"""Tools exposed to the agent.

Each section subagent picks its own subset of these (declared in
`agent_config.yaml:subagents[].tools`); the assembler has `tools: []`.
The `_TOOL_REGISTRY` at the bottom of this module is the authoritative
name → callable mapping that `agent.py:_build_section_agents` reads.

Tool inventory:
  - `search_earnings_docs`        — single-query semantic search.
  - `multi_query_search`          — batch several semantic queries at once,
                                    dedupe on `(filename, quarter, year,
                                    content[:80])` so multi-angle coverage
                                    costs one LLM tool turn instead of N.
  - `list_available_periods`      — SQL: distinct (quarter, year, filename)
                                    rows for the ticker. Lets a subagent
                                    discover the time series before drilling
                                    in.
  - `get_period_full_content`     — SQL: every page of a given (ticker,
                                    quarter, year) returned in `page_id`
                                    order. Use when semantic top-k isn't
                                    enough and the subagent needs the full
                                    filing for one period.
  - `save_report_to_volume`       — write the assembled markdown to a UC
                                    volume; called deterministically from a
                                    graph node (not exposed to subagents).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementParameterListItem, StatementState
from databricks_langchain import DatabricksVectorSearch
from langchain_core.tools import tool

from agent_server.config import load_agent_config

logger = logging.getLogger(__name__)


def _resolve_vs_client_args() -> dict:
    """Resolve a usable host + bearer token for VectorSearchClient. Works for
    all auth methods the SDK can resolve — PAT, SP OAuth M2M (Databricks Apps
    runtime), and developer OAuth U2M (`databricks auth login`).

    `databricks_langchain.DatabricksVectorSearch` only auto-passes credentials
    when `auth_type in ('pat', 'oauth-m2m')`; for `databricks-cli` (local
    dev) we feed an extracted bearer token in via `client_args` instead.
    """
    cfg = WorkspaceClient().config
    headers = cfg.authenticate()
    auth_value = headers.get("Authorization", "")
    if not auth_value.startswith("Bearer "):
        raise RuntimeError(
            "Databricks SDK could not produce a bearer token for "
            "VectorSearchClient. Configure DATABRICKS_TOKEN, DATABRICKS_CLIENT_ID/"
            "DATABRICKS_CLIENT_SECRET, or `databricks auth login --profile <p>`."
        )
    return {
        "workspace_url": cfg.host,
        "personal_access_token": auth_value.removeprefix("Bearer "),
        "disable_notice": True,
    }


@lru_cache(maxsize=1)
def _retriever() -> DatabricksVectorSearch:
    """Build the vector-search retriever once per process. The `columns`
    requested from the index come from `retrieval_config.columns` in
    `agent_config.yaml` — add a column there to surface it in tool results.

    NOTE: `page_id` and `n_elements` are intentionally absent from the
    default column list because the initial index was created with default
    `columns_to_sync` and the default subset omits some INT columns. Once
    the index is rebuilt via 06a with explicit `columns_to_sync`, those
    can be added in yaml without code changes.
    """
    cfg = load_agent_config()
    rc = cfg["retrieval_config"]
    db = cfg["databricks_config"]
    index_full_name = f"{db['catalog']}.{db['schema']}.{rc['vector_search_index']}"
    return DatabricksVectorSearch(
        endpoint=rc["vector_search_endpoint"],
        index_name=index_full_name,
        # text_column is implicit for managed-embedding Delta-sync indexes;
        # the wrapper resolves it from the index's `embedding_source_columns`.
        columns=list(rc["columns"]),
        client_args=_resolve_vs_client_args(),
    )


# Bind the default top-k from yaml at module load so the tool's schema (the
# `k` parameter the LLM sees) reflects the configured default. Section
# subagents can still override per-call by passing `k=N`; this just sets
# the fallback when they don't.
_TOP_K_DEFAULT: int = int(load_agent_config()["retrieval_config"].get("top_k", 3))


@tool
def search_earnings_docs(
    query: str,
    ticker: str,
    k: int = _TOP_K_DEFAULT,
) -> list[dict[str, Any]]:
    """Semantic search over earnings slide decks and annual reports indexed
    in Databricks Vector Search. ``ticker`` is required and used as a filter
    (cross-ticker queries are not supported in v1). Returns the top-k
    matching page chunks. Each dict contains ``content`` (the page's
    rendered markdown) plus whichever metadata columns are configured in
    ``retrieval_config.columns`` of ``agent_config.yaml``.
    """
    if not ticker:
        raise ValueError("ticker is required — pass the company's stock symbol (e.g. 'AMZN').")
    docs = _retriever().similarity_search(
        query=query,
        k=k,
        filter={"ticker": ticker.upper()},
    )
    out: list[dict[str, Any]] = []
    for d in docs:
        row: dict[str, Any] = {"content": d.page_content}
        if d.metadata:
            # Pass through every metadata field the index returned — the set
            # is determined by `retrieval_config.columns` in yaml.
            row.update(d.metadata)
        out.append(row)
    return out


@tool
def multi_query_search(
    ticker: str,
    queries: list[str],
    k_per_query: int = 2,
) -> list[dict[str, Any]]:
    """Run several semantic searches over the earnings index in one tool call,
    dedupe overlapping hits, and return a merged result list. Use when you
    have a few related angles to cover (e.g. "operating margin", "free cash
    flow", "diluted EPS") and don't want to spend one tool-call turn per
    query. ``ticker`` is required and applied as a filter to every query;
    each query returns at most ``k_per_query`` chunks. Same per-row shape as
    `search_earnings_docs` (``content`` plus the configured metadata
    columns)."""
    if not ticker:
        raise ValueError("ticker is required")
    if not queries:
        return []
    seen: set[tuple] = set()
    out: list[dict[str, Any]] = []
    for q in queries:
        for row in search_earnings_docs.invoke(
            {"query": q, "ticker": ticker, "k": k_per_query}
        ):
            key = (
                row.get("filename"),
                row.get("quarter"),
                row.get("year"),
                (row.get("content") or "")[:80],
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
    return out


# ---------------------------------------------------------------------------
# Self-querying retriever — uses an LLM to translate a natural-language
# query into a (rewritten_query, structured_filter) pair, so subagents can
# ask "AWS margin in Q4 2024" and get a filter on `(quarter, year)` for
# free instead of hoping semantic search lands on the right pages.
# ---------------------------------------------------------------------------

_DOC_CONTENTS = (
    "Excerpts from corporate earnings filings (10-Q, 10-K, slide decks): "
    "company performance commentary, financial line items, segment "
    "breakdowns, capital-return and risk discussion."
)


def _attribute_info():
    """Filterable metadata fields exposed to the query constructor.

    Keep in sync with `retrieval_config.columns` in agent_config.yaml AND
    with the index's `columns_to_sync` in
    doc_ingestion/pipeline/06a_build_index.py.
    """
    from langchain_classic.chains.query_constructor.schema import AttributeInfo

    return [
        AttributeInfo(
            name="ticker",
            description=(
                "Stock ticker symbol of the issuing company in uppercase. "
                "Allowed values: 'GOOG', 'AMZN', 'NVDA', 'MSFT'."
            ),
            type="string",
        ),
        AttributeInfo(
            name="quarter",
            description=(
                "Reporting period within a fiscal year. Use the strings "
                "'1', '2', '3', '4' for quarterly filings (10-Q), or "
                "'annual' for the full-year 10-K."
            ),
            type="string",
        ),
        AttributeInfo(
            name="year",
            description="Calendar year of the filing as an integer (e.g. 2024).",
            type="integer",
        ),
        AttributeInfo(
            name="filename",
            description="Source PDF filename. Useful when the user names a file directly.",
            type="string",
        ),
    ]


_EXAMPLES = [
    (
        "What did AMZN say about AWS operating margin in Q4 2024?",
        {
            "query": "AWS operating margin commentary",
            "filter": 'and(eq("ticker", "AMZN"), eq("quarter", "4"), eq("year", 2024))',
        },
    ),
    (
        "Summarize NVDA's full-year 2024 data center segment performance.",
        {
            "query": "data center segment revenue and growth",
            "filter": 'and(eq("ticker", "NVDA"), eq("quarter", "annual"), eq("year", 2024))',
        },
    ),
    (
        "How has GOOG's operating margin trended across 2023 and 2024?",
        {
            "query": "operating margin trend",
            "filter": 'and(eq("ticker", "GOOG"), gte("year", 2023), lte("year", 2024))',
        },
    ),
]


@lru_cache(maxsize=1)
def _self_query_components():
    """Build (query_constructor, vector_store, translator) once per process.

    Returns a tuple instead of a SelfQueryRetriever because we apply the
    hard ticker filter ourselves in `self_query_search_earnings_docs` —
    `SelfQueryRetriever._prepare_query` does `{**self.search_kwargs,
    **new_kwargs}`, so a baked-in `search_kwargs={"filter": {...}}` gets
    clobbered the moment the LLM emits any filter.

    Heavy imports (`langchain_classic`, `langchain_community`) live here so
    the cost is paid only when a subagent flips
    `use_self_querying_retriever: true` — agents that don't use the
    self-query path don't drag these into their boot path.
    """
    from databricks_langchain import ChatDatabricks
    from langchain_classic.chains.query_constructor.base import (
        load_query_constructor_runnable,
    )
    from langchain_community.query_constructors.databricks_vector_search import (
        DatabricksVectorSearchTranslator,
    )

    cfg = load_agent_config()
    sq_cfg = cfg["self_query_config"]
    llm = ChatDatabricks(endpoint=sq_cfg["model_endpoint"])
    constructor = load_query_constructor_runnable(
        llm=llm,
        document_contents=_DOC_CONTENTS,
        attribute_info=_attribute_info(),
        examples=_EXAMPLES,
        fix_invalid=True,
    )
    return constructor, _retriever(), DatabricksVectorSearchTranslator()


@tool
def self_query_search_earnings_docs(
    query: str,
    ticker: str,
    k: int = _TOP_K_DEFAULT,
) -> list[dict[str, Any]]:
    """Self-querying semantic search: an LLM auto-generates a structured
    filter (on quarter/year/filename) from your natural-language query, then
    runs hybrid retrieval over the earnings index. ``ticker`` is required
    and is always ANDed in as a hard filter, overwriting any ticker the
    auto-filter generates. Same return shape as ``search_earnings_docs``.

    Use this when your query naturally mentions a period (e.g. "AWS margin
    in Q4 2024") — the filter will pin the right filing instead of relying
    on semantic similarity alone.
    """
    if not ticker:
        raise ValueError("ticker is required")
    constructor, vector_store, translator = _self_query_components()
    structured = constructor.invoke({"query": query})
    new_query, search_kwargs = translator.visit_structured_query(structured)
    llm_filter = (search_kwargs or {}).get("filter") or {}
    # Hard-merge our ticker filter — overwrites whatever the LLM said about
    # ticker so the caller's `ticker` arg is always the source of truth.
    merged_filter = {**llm_filter, "ticker": ticker.upper()}
    docs = vector_store.similarity_search(
        query=new_query,
        k=k,
        filter=merged_filter,
        query_type="hybrid",
    )
    out: list[dict[str, Any]] = []
    for d in docs:
        row: dict[str, Any] = {"content": d.page_content}
        if d.metadata:
            row.update(d.metadata)
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# SQL-warehouse-backed tools — read directly from the source `gold_documents`
# table for fully-faithful (non-semantic) period drilling.
# ---------------------------------------------------------------------------

def _warehouse_id() -> str:
    wh = os.environ.get("DATABRICKS_WAREHOUSE_ID")
    if not wh:
        raise RuntimeError(
            "DATABRICKS_WAREHOUSE_ID is not set — bind the `sql-warehouse` "
            "Apps resource in databricks.yml or export the env var locally."
        )
    return wh


def _gold_documents_fqn() -> str:
    cfg = load_agent_config()
    db = cfg["databricks_config"]
    rc = cfg["retrieval_config"]
    return f"{db['catalog']}.{db['schema']}.{rc['gold_documents_table']}"


def _exec_sql(
    statement: str,
    parameters: list[StatementParameterListItem] | None = None,
) -> list[list[Any]]:
    resp = WorkspaceClient().statement_execution.execute_statement(
        warehouse_id=_warehouse_id(),
        statement=statement,
        parameters=parameters,
        wait_timeout="30s",
    )
    if resp.status.state != StatementState.SUCCEEDED:
        err = resp.status.error
        raise RuntimeError(
            f"Statement failed ({resp.status.state}): "
            f"{err.message if err else 'unknown'}"
        )
    return (resp.result.data_array if resp.result else None) or []


@tool
def list_available_periods(ticker: str) -> list[dict[str, Any]]:
    """List the distinct (quarter, year, filename, n_pages) tuples available
    for the ticker. Run this BEFORE drilling into a specific period so you
    know what the time series looks like (e.g. "the latest quarter is Q4
    2025; we have 8 quarters going back to Q1 2024"). Useful for the
    Overview's "latest period snapshot" decision and the Financial
    Performance time-series table."""
    if not ticker:
        raise ValueError("ticker is required")
    rows = _exec_sql(
        f"""
        SELECT quarter, year, filename, COUNT(*) AS n_pages
        FROM {_gold_documents_fqn()}
        WHERE ticker = :ticker
        GROUP BY quarter, year, filename
        ORDER BY year, quarter
        """,
        parameters=[
            StatementParameterListItem(
                name="ticker", value=ticker.upper(), type="STRING"
            )
        ],
    )
    return [
        {"quarter": r[0], "year": int(r[1]), "filename": r[2], "n_pages": int(r[3])}
        for r in rows
    ]


@tool
def get_period_full_content(
    ticker: str,
    quarter: str,
    year: int,
    max_pages: int = 30,
) -> str:
    """Pull every page of a single filing (one ticker × one quarter × one
    year) as a single concatenated markdown string, ordered by page_id.
    Use when semantic top-k isn't enough and you need the full filing for
    a specific period — e.g. building the Financial Performance table from
    a single 10-Q's complete P&L. Pages are separated by `\\n\\n---\\n\\n`.
    Truncated at ``max_pages`` to bound payload size."""
    if not ticker:
        raise ValueError("ticker is required")
    rows = _exec_sql(
        f"""
        SELECT page_id, content_md
        FROM {_gold_documents_fqn()}
        WHERE ticker = :ticker AND quarter = :quarter AND year = :year
        ORDER BY page_id
        LIMIT :max_pages
        """,
        parameters=[
            StatementParameterListItem(
                name="ticker", value=ticker.upper(), type="STRING"
            ),
            StatementParameterListItem(name="quarter", value=str(quarter), type="STRING"),
            StatementParameterListItem(name="year", value=str(int(year)), type="INT"),
            StatementParameterListItem(
                name="max_pages", value=str(int(max_pages)), type="INT"
            ),
        ],
    )
    if not rows:
        return f"_(no pages indexed for {ticker} {quarter} {year})_"
    return "\n\n---\n\n".join(str(r[1]) for r in rows)


# ---------------------------------------------------------------------------
# Save-to-volume tool (called from the `save_report` graph node, not
# exposed to subagents).
# ---------------------------------------------------------------------------

@tool
def save_report_to_volume(markdown: str, ticker: str, save_location: str) -> str:
    """Save the final research-report markdown to a Unity Catalog volume.

    Final path is ``{save_location.rstrip('/')}/{ticker}_{utc_iso}.md`` —
    e.g. ``/Volumes/.../research_reports/AMZN_20260506T143045Z.md``. The
    deployed app's service principal must have ``WRITE VOLUME`` on the
    target volume; locally, whatever auth the SDK resolves is used.

    Returns the full file path written, or ``"ERROR: <reason>"`` on
    failure (we never raise — a permission issue shouldn't fail the whole
    request when the report itself was produced successfully).
    """
    if not save_location.startswith("/Volumes/"):
        return f"ERROR: save_location must start with /Volumes/ (got {save_location!r})"
    base = save_location.rstrip("/")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    file_path = f"{base}/{ticker}_{timestamp}.md"
    try:
        WorkspaceClient().files.upload(
            file_path,
            BytesIO(markdown.encode("utf-8")),
            overwrite=True,
        )
        return file_path
    except Exception as exc:  # noqa: BLE001 — surface in saved_path, don't kill the request
        logger.warning("Failed to save report to %s: %s", file_path, exc)
        return f"ERROR: {exc}"


# ---------------------------------------------------------------------------
# Registry — maps the names used in `agent_config.yaml:subagents[].tools`
# to the actual @tool callables. `save_report_to_volume` is intentionally
# NOT in here — it's called from the `save_report` graph node, never
# exposed to a subagent's LLM loop.
# ---------------------------------------------------------------------------

SECTION_TOOL_REGISTRY: dict[str, Any] = {
    "search_earnings_docs": search_earnings_docs,
    "self_query_search_earnings_docs": self_query_search_earnings_docs,
    "multi_query_search": multi_query_search,
    "list_available_periods": list_available_periods,
    "get_period_full_content": get_period_full_content,
}
