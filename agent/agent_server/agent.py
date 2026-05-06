"""Research-report agent — classic LangGraph implementation.

The graph encodes the fixed report-generation workflow explicitly:

    START
      ↓
    extract_ticker          (1 cheap LLM call to parse the user message)
      ↓     ↘
      ↓     clarify-ticker  (short-circuit if ticker can't be determined)
      ↓        ↓
    Send×3 fan-out to write_section (parallel via langgraph.types.Send)
      ↓     ↘     ↘
      ↓      ↓     ↓
      ↓     write_section  (overview-agent | financial-performance-agent | devils-advocate-agent)
      ↘     ↙     ↙
    assemble                (1 LLM call: 3 sections → final report)
      ↓     ↘
      ↓     save_report     (only when custom_inputs.save_location is set —
      ↓        ↓             writes the markdown to a UC volume; otherwise
      ↓        ↓             this branch is skipped)
      ↘     ↙
     END

Each section subagent is a `langchain.agents.create_agent` with only
`[search_earnings_docs]` as its tool — no agent-harness overhead, no
`write_todos` / virtual-filesystem schemas shipped on every model turn.
The yaml at `agent/agent_config.yaml` is the source of truth for prompts;
this module wires the four entries (three section subagents + assembler)
into the graph.
"""

from __future__ import annotations

import logging
import operator
import os
from typing import Annotated, AsyncGenerator, TypedDict

import litellm
import mlflow
from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent
from langchain.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
)

from agent_server.config import load_agent_config
from agent_server.tools import SECTION_TOOL_REGISTRY, save_report_to_volume
from agent_server.utils import get_session_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------

def _configure_mlflow() -> None:
    """Pin tracking URI + experiment + enable LangChain autolog at module
    load so every invocation in this process logs to the same place.

    Resolution order for the experiment:
      1. `MLFLOW_EXPERIMENT_ID` env var — injected by Databricks Apps via the
         `valueFrom: experiment` resource binding in `databricks.yml`. Lets
         the bundle manage dev/prod experiment separation (dev-mode prefixes
         paths automatically; prod uses the unprefixed name).
      2. `agent_config.yaml:agent_config.mlflow_experiment_name` — fallback
         for local dev / non-bundled runs. `mlflow.set_experiment()`
         auto-creates the path if it doesn't exist.

    Tracking URI is explicitly set to `databricks` so workspace paths
    resolve correctly even if `MLFLOW_TRACKING_URI` isn't injected.
    """
    mlflow.set_tracking_uri("databricks")
    if not os.environ.get("MLFLOW_EXPERIMENT_ID"):
        config = load_agent_config()
        experiment_name = (config.get("agent_config") or {}).get("mlflow_experiment_name")
        if experiment_name:
            try:
                mlflow.set_experiment(experiment_name)
            except Exception:  # noqa: BLE001 — surface as a warning, don't block startup
                logger.warning(
                    "Failed to set MLflow experiment %r; falling back to default.",
                    experiment_name,
                    exc_info=True,
                )
    mlflow.langchain.autolog()


_configure_mlflow()
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)
litellm.suppress_debug_info = True


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class ReportState(TypedDict, total=False):
    """State that flows through the report-generation graph.

    `sections` uses `operator.add` as its reducer so the three parallel
    `write_section` branches can each return a single-element list and the
    parent state ends up with a list of length 3 by the time `assemble`
    runs.

    `save_location` is optional: when set (via `custom_inputs.save_location`
    on the inbound `/invocations` request), the graph routes through
    `save_report` after `assemble`; when absent, that node is skipped.
    `saved_path` is the result of the save (a real volume path on success
    or an `"ERROR: ..."` string on failure).
    """

    user_message: str
    ticker: str
    sections: Annotated[list[dict], operator.add]
    final_report: str
    save_location: str
    saved_path: str


class SectionTask(TypedDict):
    """The per-branch input dispatched via Send to `write_section`."""

    ticker: str
    section_name: str  # "overview" | "financial" | "devils_advocate"


# ---------------------------------------------------------------------------
# Build the section + assembler subagents from the existing yaml config —
# we reuse all four prompts unchanged so prompt iteration stays in yaml.
# ---------------------------------------------------------------------------

_CONFIG = load_agent_config()
_SUBAGENTS_BY_NAME = {entry["name"]: entry for entry in _CONFIG.get("subagents", [])}

# Map our internal section identifier → yaml subagent name.
_SECTION_TO_AGENT_NAME = {
    "overview": "overview-agent",
    "financial": "financial-performance-agent",
    "devils_advocate": "devils-advocate-agent",
}

# Canonical order for the assembler's input.
_SECTION_ORDER = ["overview", "financial", "devils_advocate"]
_SECTION_HEADERS = {
    "overview": "## Section 1 — Overview & Business Segments",
    "financial": "## Section 2 — Financial Performance & Shareholder Returns",
    "devils_advocate": "## Section 3 — Devil's Advocate",
}


def _resolve_tools(entry: dict) -> list:
    """Translate the yaml `tools:` list (a list of registry names) into
    actual @tool callables. Defaults to `[search_earnings_docs]` for
    backwards compat — section subagents that omit `tools:` still get the
    base retrieval tool. An empty list (`tools: []`) means no tools."""
    if "tools" not in entry:
        return [SECTION_TOOL_REGISTRY["search_earnings_docs"]]
    return [SECTION_TOOL_REGISTRY[name] for name in entry["tools"]]


def _build_section_agents() -> dict:
    """Build a `create_agent` per section subagent. Each one carries only
    its own system prompt + the tools its yaml entry declares — no
    agent-harness overhead on every model turn."""
    out = {}
    for section_name, agent_name in _SECTION_TO_AGENT_NAME.items():
        entry = _SUBAGENTS_BY_NAME[agent_name]
        out[section_name] = create_agent(
            model=ChatDatabricks(endpoint=entry["model_endpoint"]),
            tools=_resolve_tools(entry),
            system_prompt=entry["system_prompt"],
        )
    return out


_SECTION_AGENTS = _build_section_agents()
_ASSEMBLER_ENTRY = _SUBAGENTS_BY_NAME["report-assembler-agent"]


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

# Tiny model used only to extract the ticker from the user message.
_TICKER_EXTRACTOR = ChatDatabricks(endpoint="databricks-claude-haiku-4-5")
_TICKER_EXTRACTION_PROMPT = (
    "Extract the stock ticker the user is asking about. "
    "Map company names: Alphabet→GOOG, Amazon→AMZN, Nvidia→NVDA, Microsoft→MSFT. "
    "Respond with ONLY the ticker symbol in uppercase (e.g. 'AMZN'). "
    "If you cannot determine a single ticker, respond with 'UNKNOWN'."
)


async def extract_ticker(state: ReportState) -> dict:
    user_msg = state.get("user_message", "")
    response = await _TICKER_EXTRACTOR.ainvoke(
        [
            {"role": "system", "content": _TICKER_EXTRACTION_PROMPT},
            {"role": "user", "content": user_msg},
        ]
    )
    ticker = (response.content or "").strip().upper()
    return {"ticker": ticker}


async def write_section(task: SectionTask) -> dict:
    """Invoke the section subagent for the given ticker. Returns a
    single-element list so the parent state's `sections` accumulator
    (operator.add reducer) appends one entry per parallel branch."""
    section_name = task["section_name"]
    ticker = task["ticker"]
    agent = _SECTION_AGENTS[section_name]
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Write your section for ticker {ticker}.",
                }
            ],
        }
    )
    final_msg = result["messages"][-1]
    section_md = (
        final_msg.content
        if hasattr(final_msg, "content") and isinstance(final_msg.content, str)
        else str(final_msg)
    )
    return {"sections": [{"name": section_name, "content": section_md}]}


async def assemble(state: ReportState) -> dict:
    """Combine the three section outputs + ticker into the final report by
    calling the assembler subagent once. The assembler has no tools — it
    formats the title, passes sections through verbatim, and adds the
    Synthesis paragraph."""
    by_name = {s["name"]: s["content"] for s in state.get("sections", [])}
    parts = [f"Ticker: {state['ticker']}"]
    for name in _SECTION_ORDER:
        parts.append(_SECTION_HEADERS[name])
        parts.append(by_name.get(name, "_(no data returned for this section)_"))
    user_payload = "\n\n".join(parts)

    assembler_model = ChatDatabricks(endpoint=_ASSEMBLER_ENTRY["model_endpoint"])
    response = await assembler_model.ainvoke(
        [
            {"role": "system", "content": _ASSEMBLER_ENTRY["system_prompt"]},
            {"role": "user", "content": user_payload},
        ]
    )
    return {"final_report": response.content or ""}


async def clarify(state: ReportState) -> dict:
    """Short-circuit when ticker extraction couldn't resolve a single ticker."""
    return {
        "final_report": (
            "I couldn't determine which ticker you'd like a report on. "
            "Please specify one of: GOOG, AMZN, NVDA, MSFT."
        )
    }


async def save_report(state: ReportState) -> dict:
    """Persist the final markdown to the caller-supplied UC volume path.
    Runs only when `save_location` is set on the state (driven by the
    conditional edge below). The deterministic-node design keeps the LLM
    out of the save decision — when `save_location` is set, the save
    happens; when it isn't, this node never runs.
    """
    path = save_report_to_volume.invoke(
        {
            "markdown": state["final_report"],
            "ticker": state["ticker"],
            "save_location": state["save_location"],
        }
    )
    footer = f"\n\n---\n_Saved to: `{path}`_"
    return {
        "saved_path": path,
        "final_report": state["final_report"] + footer,
    }


def route_after_assemble(state: ReportState):
    """Conditional edge after `assemble`: route to `save_report` only if
    the caller passed a UC volume path via `custom_inputs.save_location`.
    Anything that doesn't look like a `/Volumes/...` path skips the save."""
    loc = state.get("save_location") or ""
    return "save_report" if loc.startswith("/Volumes/") else END


def route_after_ticker(state: ReportState):
    """Conditional edge: fan out to three parallel section writers OR
    short-circuit to the clarification node. Returning a list of `Send`
    objects is LangGraph's idiom for parallel dispatch."""
    ticker = state.get("ticker")
    if not ticker or ticker == "UNKNOWN":
        return "clarify"
    return [
        Send("write_section", {"ticker": ticker, "section_name": name})
        for name in _SECTION_ORDER
    ]


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------

def _build_graph():
    g = StateGraph(ReportState)
    g.add_node("extract_ticker", extract_ticker)
    g.add_node("write_section", write_section)
    g.add_node("assemble", assemble)
    g.add_node("save_report", save_report)
    g.add_node("clarify", clarify)

    g.add_edge(START, "extract_ticker")
    g.add_conditional_edges(
        "extract_ticker",
        route_after_ticker,
        ["write_section", "clarify"],
    )
    g.add_edge("write_section", "assemble")
    g.add_conditional_edges(
        "assemble",
        route_after_assemble,
        ["save_report", END],
    )
    g.add_edge("save_report", END)
    g.add_edge("clarify", END)
    return g.compile()


_GRAPH = None


def init_agent():
    """Lazy-build + cache the compiled graph."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


# ---------------------------------------------------------------------------
# MLflow ResponsesAgent handlers
# ---------------------------------------------------------------------------

@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    outputs = [
        event.item
        async for event in stream_handler(request)
        if event.type == "response.output_item.done"
    ]
    return ResponsesAgentResponse(output=outputs)


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    if session_id := get_session_id(request):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    graph = init_agent()
    user_message = next(
        (
            i.content
            for i in request.input
            if hasattr(i, "content") and isinstance(i.content, str)
        ),
        "",
    )
    initial_state: ReportState = {"user_message": user_message, "sections": []}

    # Optional persistence: callers can pass `custom_inputs.save_location`
    # (a `/Volumes/...` path) to have the assembled markdown also written
    # to UC volumes. Absence of the field means no save.
    if request.custom_inputs and isinstance(request.custom_inputs, dict):
        if save_location := request.custom_inputs.get("save_location"):
            initial_state["save_location"] = save_location

    # Run the graph to completion, then emit the final report as a single
    # text output. True per-token streaming would require switching the
    # assembler call to `.astream()` and forwarding the message chunks
    # through `process_agent_astream_events` — left for a follow-up.
    final_state = await graph.ainvoke(initial_state)
    final_message = AIMessage(content=final_state.get("final_report", ""))
    for item in output_to_responses_items_stream([final_message]):
        yield item
