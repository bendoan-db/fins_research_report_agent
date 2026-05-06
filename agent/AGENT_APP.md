# `agent` — equity research-report agent

A Databricks-deployed agent that produces a structured **markdown research report** for a publicly traded company on demand. The user issues a natural-language request like *"Research report for AMZN"* and the agent returns a three-section report (Overview, Financial Performance, Devil's Advocate) grounded in earnings slide decks and annual reports indexed in Databricks Vector Search.

---

## Overview

**What it is.** A FastAPI server backed by a LangGraph `StateGraph`, deployed as a Databricks App. It exposes the standard MLflow ResponsesAgent `/invocations` endpoint. Each section is authored by a specialist subagent (`langchain.agents.create_agent`) with its own toolkit; an assembler glues their outputs together with a closing Synthesis paragraph.

**What it reads.** The `doan.difficult_doc_qa.gold_documents` table (~640 page-grain rows) built by the `doc_ingestion` pipeline, exposed for retrieval via the `gold_documents_index` Vector Search index on the `financebench` endpoint.

**Why classic LangGraph (not deepagents).** The workflow is a fixed DAG — the same three subagents in the same order, every time. There's no decision the orchestrator makes at runtime that justifies the deepagents harness's per-call token overhead. Each section subagent's call carries only its own system prompt + the schemas of the tools it actually uses, not 10K+ tokens of `write_todos` + virtual filesystem + `task` schemas. That's what lets parallel fan-out fit within the workspace's `databricks-claude-sonnet-4-6` per-minute token budget.

---

## Agent logic

```
chat message ─▶ AgentServer.stream_handler ─▶ init_agent() ─▶ compiled StateGraph
                                                                       │
                                                                     START
                                                                       │
                                                              ┌────────┴────────┐
                                                              │ extract_ticker  │  haiku — parse ticker
                                                              └────────┬────────┘
                                                                       │
                                                          route_after_ticker (conditional edge)
                                                            │                │
                                                  ticker valid          ticker == UNKNOWN
                                                            │                │----
                                                  Send×3 (parallel)          ┌───┴───--┐
                                          ┌─────────────────┼─────────────┐  │ clarify │  short-circuit msg
                                          ▼                 ▼             ▼  └──--───--┘
                                ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
                                │ write_section   │ │ write_section   │ │ write_section   │  3× same node
                                │ overview        │ │ financial       │ │ devils_advocate │  parallel
                                │ + segments      │ │ + capital ret.  │ │                 │  branches
                                └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
                                         │                   │                   │
                                         └───────────────────┴───────────────────┘
                                                             │
                                            sections accumulator (operator.add)
                                                             │
                                                      ┌──────┴──────┐
                                                      │   assemble  │  haiku — title + 3 sections + Synthesis
                                                      └──────┬──────┘
                                                             │
                                                  route_after_assemble
                                                       │            │
                                                save_location set?  no → END
                                                       │
                                                       ▼
                                                ┌──────────────┐
                                                │  save_report │  optional — writes markdown to a UC volume
                                                └──────┬───────┘
                                                       │
                                                      END
                                                       │
                                                  chat reply
```

**Per-node summary**:

| Node | Implementation | Purpose |
|---|---|---|
| `extract_ticker` | one `ChatDatabricks(haiku-4.5).ainvoke(...)` | Resolve the ticker from the user message; return `"UNKNOWN"` if unclear. |
| `clarify` | pure Python (no LLM) | Short-circuits the graph with a fixed reply asking the user to specify a ticker. |
| `write_section` (×3 in parallel) | `langchain.agents.create_agent(...)` per subagent | Each section subagent runs its own tool-calling loop, returns markdown for its section. |
| `assemble` | one `ChatDatabricks(haiku-4.5).ainvoke(...)` | Inlines the three section markdowns + ticker, returns the final report (title + verbatim sections + closing `## Synthesis` paragraph). |
| `save_report` | deterministic, calls `save_report_to_volume.invoke(...)` | Optional — only runs when the request includes `custom_inputs.save_location`. Persists the markdown to a UC volume; appends a footer line `_Saved to: <path>_` to the response. |

**Why the bookend nodes use Haiku, not Sonnet**: `extract_ticker` and `assemble` are single-purpose calls (parse one ticker / concatenate three sections + write one paragraph). They don't benefit from Sonnet's reasoning depth, and using Haiku keeps the per-minute token budget on Sonnet reserved for the section subagents that actually need it.

**Per-subagent toolkits** (declared in `agent_config.yaml`, resolved by `agent.py:_resolve_tools` against the `SECTION_TOOL_REGISTRY` in `agent_server/tools.py`):

| Subagent | Tools |
| --- | --- |
| `overview-agent` | `search_earnings_docs`, `multi_query_search`, `list_available_periods` |
| `financial-performance-agent` | `search_earnings_docs`, `list_available_periods`, `get_period_full_content` |
| `devils-advocate-agent` | `search_earnings_docs`, `multi_query_search` |
| `report-assembler-agent` | `tools: []` (formatting only) |

---

## Key files

```
agent/
├── AGENT_APP.md                    ← this file
├── agent_config.yaml               ← single source of truth for prompts + endpoints +
│                                     subagent toolkits + retrieval params + experiment path
├── app.yaml                        ← Databricks Apps manifest (command + env bindings)
├── pyproject.toml                  ← deps (langgraph, langchain, databricks-langchain, mlflow, …)
├── requirements.txt                ← mirror of pyproject deps for the test_agent notebook's %pip
├── .env.example                    ← local-dev env vars (DATABRICKS_CONFIG_PROFILE etc.)
├── .gitignore                      ← excludes .venv, *.db, .env from the bundle source snapshot
├── agent_server/
│   ├── __init__.py
│   ├── agent.py                    ← StateGraph wiring, MLflow setup, ResponsesAgent handlers,
│   │                                 _resolve_tools registry lookup, save_report node
│   ├── config.py                   ← lru_cache'd load_agent_config()
│   ├── tools.py                    ← @tool callables: search_earnings_docs,
│   │                                 multi_query_search, list_available_periods,
│   │                                 get_period_full_content, save_report_to_volume +
│   │                                 SECTION_TOOL_REGISTRY name→callable map
│   ├── start_server.py             ← MLflow AgentServer entry point ("uv run start-server")
│   └── utils.py                    ← session-id helper for trace metadata
└── scripts/
    ├── __init__.py
    └── test_agent.py               ← Databricks notebook for in-process smoke testing —
                                      renders the report inline + saves a copy to a volume
```

The bundle definition + the app's resource bindings live in the **root** `databricks.yml`, not in this directory — see Deployment below.

---

## Deployment with Databricks Apps

The agent is deployed via the project bundle alongside the gallery app. Bundle deploy creates the Databricks App (with its service principal), the MLflow experiment, and all the resource bindings the runtime needs.

### Bundle resources declared in root `databricks.yml`

```yaml
resources:
  experiments:
    research_agent_experiment:    # bundle-managed, dev-mode prefixed in dev target
      name: ${var.research_agent_experiment_path}

  apps:
    research_agent:
      name: research-agent
      source_code_path: ./agent
      resources:
        - experiment              ← MLFLOW_EXPERIMENT_ID via valueFrom
        - section-endpoint        ← databricks-claude-sonnet-4-6 (CAN_QUERY)
        - orchestration-endpoint  ← databricks-claude-haiku-4-5 (CAN_QUERY)
        - sql-warehouse           ← DATABRICKS_WAREHOUSE_ID via valueFrom (CAN_USE)
```

`app.yaml` then maps each binding into an env var the agent runtime reads:

```yaml
env:
  - { name: MLFLOW_TRACKING_URI,  value: databricks }
  - { name: MLFLOW_REGISTRY_URI,  value: databricks-uc }
  - { name: MLFLOW_EXPERIMENT_ID, valueFrom: experiment }
  - { name: DATABRICKS_WAREHOUSE_ID, valueFrom: sql-warehouse }
```

The Apps runtime auto-injects `DATABRICKS_HOST` + `DATABRICKS_CLIENT_ID` + `DATABRICKS_CLIENT_SECRET` (the SP's OAuth M2M credentials) and the framework port `DATABRICKS_APP_PORT` (8000). Anything `WorkspaceClient()` does — Vector Search, files API, statement execution — goes through that SP.

### One-time out-of-band IAM grants

These can't be expressed as Apps resource bindings (the bundle DSL only exposes `experiment` / `serving_endpoint` / `sql_warehouse` / `uc_securable` / a few others — and `uc_securable` doesn't cover catalogs, schemas, or Vector Search resources). After first deploy, get the SP id (`databricks apps get research-agent --profile DEFAULT --output json | jq -r .service_principal_client_id`) and run:

```sql
-- Catalog/schema USE so the SP can resolve `doan.difficult_doc_qa.*`
GRANT USE CATALOG ON CATALOG doan TO `<sp_client_id>`;
GRANT USE SCHEMA  ON SCHEMA  doan.difficult_doc_qa TO `<sp_client_id>`;

-- Vector Search index (queried via `search_earnings_docs` /
-- `multi_query_search`)
GRANT USE ON VECTOR SEARCH ENDPOINT financebench TO `<sp_client_id>`;
GRANT READ ON VECTOR SEARCH INDEX doan.difficult_doc_qa.gold_documents_index
  TO `<sp_client_id>`;

-- Source table (queried via `list_available_periods` /
-- `get_period_full_content` through the SQL warehouse)
GRANT SELECT ON TABLE doan.difficult_doc_qa.gold_documents
  TO `<sp_client_id>`;

-- Optional: enable the per-request `custom_inputs.save_location` save tool.
-- Both READ and WRITE are required (the SDK reads volume metadata before
-- writing); a write-only grant fails with "does not have READ VOLUME".
CREATE VOLUME IF NOT EXISTS doan.difficult_doc_qa.research_reports;
GRANT READ VOLUME, WRITE VOLUME
  ON VOLUME doan.difficult_doc_qa.research_reports
  TO `<sp_client_id>`;
```

The same statements are inlined as a comment on the `research_agent` resource in `databricks.yml` so future maintainers don't have to re-derive them.

### Deploy + run

```bash
# 1. Make sure the Vector Search index exists and is ONLINE — created by the
#    ingestion pipeline's 06a_build_index step.
databricks vector-search-indexes get-index doan.difficult_doc_qa.gold_documents_index --profile DEFAULT

# 2. Validate + deploy the bundle (creates the app + experiment + bindings).
databricks bundle validate --profile DEFAULT
databricks bundle deploy   --profile DEFAULT

# 3. (First deploy only) apply the SQL grants above using the SP id printed
#    by:  databricks apps get research-agent --profile DEFAULT

# 4. Start (or restart) the FastAPI server.
databricks bundle run research_agent --profile DEFAULT
# → "App started successfully"
# → URL: https://research-agent-<workspace-id>.aws.databricksapps.com
```

### Test the deployed endpoint

```bash
TOKEN=$(databricks auth token --profile DEFAULT --output json | jq -r .access_token)
APP_URL=https://research-agent-2309167578215964.aws.databricksapps.com

# Without save:
curl -X POST "${APP_URL}/invocations" \
  -H "Authorization: Bearer ${TOKEN}" -H "Content-Type: application/json" \
  -d '{"input":[{"role":"user","content":"Research report for NVDA"}]}'

# With save to a UC volume:
curl -X POST "${APP_URL}/invocations" \
  -H "Authorization: Bearer ${TOKEN}" -H "Content-Type: application/json" \
  -d '{
    "input":[{"role":"user","content":"Research report for NVDA"}],
    "custom_inputs":{"save_location":"/Volumes/doan/difficult_doc_qa/research_reports"}
  }'
```

A typical end-to-end response is ~90–105 seconds and 12–16 KB of markdown. Every graph node, subagent call, and tool invocation lands as a span in the configured MLflow experiment.

### Local devloop

The notebook at `agent/scripts/test_agent.py` runs the agent in-process, which is faster to iterate on than redeploying. After `databricks bundle deploy --profile DEFAULT`, open it at `/Workspace/Users/<me>/.bundle/difficult-doc-qa/dev/files/agent/scripts/test_agent` in the workspace, attach to serverless, change the `question` (and optionally `save_location`) at the top, Run All. Renders the final markdown via `IPython.display.Markdown` and prints the elapsed time + experiment id.

---

## Source data + vector index

`doan.difficult_doc_qa.gold_documents` (~640 rows after the latest pipeline run; one row per page):

```
path             STRING       -- /Volumes/.../earnings_slides/<file>
filename         STRING
ticker           STRING       -- 'GOOG' | 'AMZN' | 'NVDA' | 'MSFT'
quarter          STRING       -- '1'..'4' or 'annual'
year             INT
page_id          INT
uuid             STRING       -- stable page id (UUID v4) generated in step 04
n_elements       INT
content_md       STRING       -- markdown render of every element on the page
image_uri        STRING       -- /Volumes/.../ai_parse_doc_images/<...>.jpg
content_to_embed STRING       -- content_md + appended '## Summary' from step 05
```

The Vector Search index `doan.difficult_doc_qa.gold_documents_index`:

| Setting             | Value                                        |
| ------------------- | -------------------------------------------- |
| Endpoint            | `financebench`                               |
| Sync mode           | Delta-sync (managed embeddings)              |
| Embedding model     | `databricks-qwen3-embedding-0-6b`            |
| Source table        | `doan.difficult_doc_qa.gold_documents`       |
| Primary key         | `uuid`                                       |
| Embedding source    | `content_to_embed`                           |
| Columns synced      | `content_to_embed`, `ticker`, `quarter`, `year`, `filename`, `image_uri`, `path` |

Index creation lives in `doc_ingestion/pipeline/06a_build_index.py` — the agent only consumes the index, doesn't build it.

---

## Configuration in `agent/agent_config.yaml`

```yaml
databricks_config:
  catalog: doan
  schema: difficult_doc_qa

retrieval_config:
  vector_search_endpoint: financebench
  vector_search_index: gold_documents_index
  embedding_model: databricks-qwen3-embedding-0-6b
  gold_documents_table: gold_documents
  top_k: 5                              # default k for search_earnings_docs (LLM can override per-call)
  columns: [ticker, quarter, year, filename]   # metadata fields requested + returned

agent_config:
  orchestrator_endpoint: databricks-claude-haiku-4-5    # used by extract_ticker
  report_title_template: "# Research report — {ticker}"
  mlflow_experiment_name: /Workspace/Users/.../equity-research-agent

subagents:
  - name: overview-agent                  # Overview + Business Segments
    model_endpoint: databricks-claude-sonnet-4-6
    tools: [search_earnings_docs, multi_query_search, list_available_periods]
    system_prompt: |  # (~1K chars; defines the section structure + tool guidance)
      ...
  - name: financial-performance-agent     # P&L + Equity & Shareholder Returns
    tools: [search_earnings_docs, list_available_periods, get_period_full_content]
    ...
  - name: devils-advocate-agent
    tools: [search_earnings_docs, multi_query_search]
    ...
  - name: report-assembler-agent          # tools: [] (formatting only)
    model_endpoint: databricks-claude-haiku-4-5
    tools: []
    ...
```

Yaml is the single source of truth. `agent_server/config.py` loads it through `lru_cache` so every module that needs config (agent.py, tools.py) goes through one helper.

---

## Tool inventory

All tools are defined in `agent/agent_server/tools.py`. The `SECTION_TOOL_REGISTRY` at the bottom of that file is the name → callable map that `agent.py:_resolve_tools` reads when wiring each subagent.

- **`search_earnings_docs(query, ticker, k)`** — single-query semantic search. `k` defaults to `retrieval_config.top_k` from yaml; result rows always carry `content` plus the metadata columns configured in `retrieval_config.columns`. Uses `databricks_langchain.DatabricksVectorSearch` against `financebench/gold_documents_index`.
- **`multi_query_search(ticker, queries, k_per_query)`** — batches several semantic queries in one tool call, dedupes overlapping hits on `(filename, quarter, year, content[:80])`. Saves N LLM tool turns when a subagent has multiple related angles to cover (e.g. Devil's Advocate's standard risk-keyword sweep).
- **`list_available_periods(ticker)`** — SQL against `gold_documents`: returns distinct `(quarter, year, filename, n_pages)` rows, ordered. Lets a subagent discover the time series before drilling in.
- **`get_period_full_content(ticker, quarter, year, max_pages=30)`** — SQL against `gold_documents`: every page of one filing in `page_id` order, joined with `\n\n---\n\n`. Use when semantic top-k isn't enough and the subagent needs the full filing for a single period.
- **`save_report_to_volume(markdown, ticker, save_location)`** — writes the assembled markdown to a UC volume at `{save_location}/{ticker}_{utc_iso}.md`. Called deterministically from the `save_report` graph node; **not** in `SECTION_TOOL_REGISTRY` so subagents can't invoke it.

The two SQL-backed tools (`list_available_periods`, `get_period_full_content`) are why the bundle binds a `sql-warehouse` Apps resource — they execute parameterized statements via `WorkspaceClient().statement_execution.execute_statement(warehouse_id=os.environ["DATABRICKS_WAREHOUSE_ID"], ...)`.

---

## Optional save to Unity Catalog volume

When the request includes `custom_inputs.save_location`, the graph routes through a `save_report` node that persists the assembled markdown to a UC volume:

```json
{
  "input": [{"role": "user", "content": "Research report for AMZN"}],
  "custom_inputs": {"save_location": "/Volumes/doan/difficult_doc_qa/research_reports"}
}
```

**Filename pattern**: `{save_location.rstrip('/')}/{TICKER}_{utc_iso}.md` — e.g. `/Volumes/doan/difficult_doc_qa/research_reports/AMZN_20260506T143045Z.md` (UTC, filename-safe ISO 8601 with no colons).

**Decision logic**: a `route_after_assemble` conditional edge checks `state.get("save_location")`. If it's a non-empty string starting with `/Volumes/`, the graph routes to `save_report` then `END`; otherwise straight to `END`. The LLM never decides — the routing is data-driven from `custom_inputs`. Choosing a deterministic node over an LLM-exposed `@tool` keeps the assembler's "your entire response IS the final report" contract clean and prevents the model from forgetting/hallucinating the save call.

**Caller-visible signal**: when the save runs, a footer line `_Saved to: <path>_` is appended to the markdown body so callers see the saved location without needing to parse `custom_outputs`. On failure (e.g. missing `WRITE VOLUME` grant on the SP), the tool returns `"ERROR: ..."` instead of raising — the report is still produced, the error string just appears in the footer for visibility.

**Required IAM grants**: the SP needs **both** `READ VOLUME` and `WRITE VOLUME` on the destination volume (the SDK's `files.upload` does a metadata read before writing; a write-only grant fails with `"User does not have READ VOLUME on Volume ..."`). See the inline grant block in the Deployment section.

---

## Open questions / known limitations

1. **Source semantic mismatch** — the corpus is earnings *slide decks* + NVDA *annual reports*, not literal earnings call transcripts. Reports reflect what's in the documents (summarized financial highlights and management commentary, not Q&A transcripts). Adding true call transcripts is a pipeline-side change; the agent picks them up automatically once `gold_documents` includes them.
2. **Period scope** — by default each section subagent retrieves across all quarters/years in the index for the chosen ticker (filter is on `ticker`, not period). If the user says *"for Q3 2025"*, that text is in the user message but the LangGraph flow doesn't currently propagate it as a retrieval filter — section subagents see only the ticker. Fixable by either parsing the period in `extract_ticker` and threading into `write_section`, or extending `search_earnings_docs` to accept optional `quarter` / `year` filters.
3. **Subagent model selection** — section subagents default to `databricks-claude-sonnet-4-6`; the bookend nodes (extract_ticker, assemble) and the assembler subagent use `databricks-claude-haiku-4-5`. Devil's Advocate might benefit from a stronger reasoning model (e.g. `databricks-claude-opus-4-1`) — flagged for later tuning, easy yaml-only change.
4. **Citation rendering** — citations land as inline parentheticals `(filename)` (page numbers are absent because the initial Vector Search index was built with default `columns_to_sync` and silently dropped INT columns). When 06a is rerun with explicit `columns_to_sync` including `page_id`, the prompts can switch back to `(filename, page N)`.
