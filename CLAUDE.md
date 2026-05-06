# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Databricks Asset Bundle (`databricks.yml` → bundle name `difficult-doc-qa`) with three top-level pieces:

- `doc_ingestion/` — medallion-style ingestion pipeline over earnings slide decks (PDF/PPTX) producing a Vector Search index.
- `agent/` — LangGraph research-report agent (Databricks App) over that index. Three parallel section subagents + an assembler, served via MLflow's `ResponsesAgent`.
- `doc_gallery_app/` — Streamlit Databricks App that renders parsed pages side-by-side with their extracted elements (page image + element overlays + figure captions) from `gold_documents_gallery`.

Per-piece config files (edit these rather than hardcoding values):

- `doc_ingestion/ingestion_config.yaml` — pipeline target tables, catalog/schema/volume names, captioning endpoint + prompt, ticker allow-list, vector-index settings.
- `agent/agent_config.yaml` — subagent system prompts, model endpoints, per-subagent `tools:` lists, retrieval columns, MLflow experiment name.

## Pipeline shape (run in order)

1. `doc_ingestion/setup/load_files_to_volume.py` — uploads everything in `doc_ingestion/setup/data/` into `/Volumes/{catalog}/{schema}/{volume}` via the SDK Files API. Idempotent (skips files whose size already matches).
2. `doc_ingestion/pipeline/01_extract_metadata.py` — Auto Loader (`cloudFiles` over `binaryFile`) streams new `*.{pdf,pptx}` from the volume into `bronze_raw_documents_with_metadata`. Filename-derived `ticker`/`quarter`/`year` come from a regex-first `COALESCE`, with `ai_classify` v2 / `ai_extract` v2 as fallback (Spark short-circuits, so AI cost is paid only on regex miss).
3. `doc_ingestion/pipeline/02_parse_documents.py` — three sequential `Trigger.AvailableNow` streams off the metadata table: full parsed VARIANT → `silver_document_raw_parsed_outputs`; exploded pages → `silver_document_pages` (with `image_uri` written to a separate `ai_parse_doc_images` volume); exploded elements → `silver_document_elements` (each element row carries the `page_id` hoisted from `bbox[0].page_id`).
4. `doc_ingestion/pipeline/03_caption_figures.py` — streams figure-typed elements, joins to `silver_document_pages` on `(path, page_id)` for the page image URI, runs a Pillow UDF (`crop_image_to_png`) over the FUSE mount, and pipes the cropped PNG into `ai_query(caption_endpoint, caption_prompt, files => ...)`. Output: `silver_figure_captions`.
5. `doc_ingestion/pipeline/04_aggregate_pages.py` — streams `silver_document_elements` (`Trigger.AvailableNow`), in `foreachBatch` static-reads `silver_figure_captions` for the affected `(path, page_id)` keys, joins on `(path, element_id)`, and `MERGE INTO`s a one-row-per-page table (`silver_document_elements_aggregated`) with a `content_md` column that renders elements as markdown blocks (`title`→`#`, `section_header`→`##`, `figure`→`### Figure` + caption, `table`→`### Table` + content, others→raw content). Must run **after** 03 so caption rows are present at recompute time.
6. `doc_ingestion/pipeline/05_generate_summaries.py` — streams `silver_document_elements_aggregated` (`Trigger.AvailableNow`), static-joins `silver_document_pages` on `(path, page_id)` for `image_uri`, and calls `ai_query(summary_endpoint, prompt + ticker/period + content_md, failOnError => false)` inline to produce a per-page summary. A second `selectExpr` derives `content_to_embed = coalesce(concat(content_md, '\n\n## Summary\n\n', summary), content_md)` so a NULL summary falls back to the raw content. Output: `gold_documents` with `image_uri` and `content_to_embed` (single field for the downstream embedding step). Must run **after** 04.
7. `doc_ingestion/pipeline/06a_build_index.py` — idempotent setup that creates (or syncs) the Databricks Vector Search index `gold_documents_index` over `gold_documents` with managed embeddings (`databricks-qwen3-embedding-0-6b`). Primary key = `uuid`, embedding source = `content_to_embed`. Configured under `vector_index_step` in `ingestion_config.yaml`. Must run **after** 05; consumed by the research agent in `agent/`.
8. `doc_ingestion/pipeline/06b_build_gallery.py` — streams `gold_documents` (`Trigger.AvailableNow`) and stream-static-joins a per-page elements array built from `silver_document_elements` left-joined to `silver_figure_captions` on `(path, element_id)`, grouped by `(path, page_id)` with `sort_array(collect_list(struct(...)))`. Output: `gold_documents_gallery` with `image_uri`, `content_md`, a `summary` column derived by stripping the `## Summary` header from `content_to_embed`, and an `elements ARRAY<STRUCT<element_id, type, content, description, bbox VARIANT, caption>>` payload designed for a UI gallery that overlays element rectangles on the page image and shows figure captions inline. Must run **after** 05.

Each streaming write uses a per-table checkpoint at `/Volumes/.../<checkpoint_prefix>/<table_name>`, where the prefix is configured per stream in `ingestion_config.yaml` (e.g. `metadata_extraction_step.checkpoint_prefix`, `document_parsing_step.parsed_document_table_pages_checkpoint_prefix`, `figure_captioning_step.caption_checkpoint_prefix`). **To force reprocessing, bump the prefix value (e.g. `_checkpoints/v1` → `_checkpoints/v2`) AND drop the affected rows** — Auto Loader is path-keyed and the `Trigger.AvailableNow` streams will otherwise treat the source as already consumed. Notebook 02 writes three tables from one source via three independent prefixes; reprocessing typically means bumping all three together.

## Agent shape (`agent/`)

A classic LangGraph `StateGraph` (no agent-harness overhead — explicitly avoids `write_todos` / virtual-filesystem schemas). Wired in `agent/agent_server/agent.py`:

```
START → extract_ticker → (Send×3 fan-out) → write_section ×3 → assemble → [save_report?] → END
                       ↘ clarify (when ticker can't be resolved) → END
```

- `extract_ticker` — one cheap `databricks-claude-haiku-4-5` call to map company name → ticker.
- `write_section` — three parallel branches dispatched via `langgraph.types.Send`. Each branch is a `langchain.agents.create_agent` built from a yaml entry in `agent_config.yaml:subagents` (`overview-agent`, `financial-performance-agent`, `devils-advocate-agent`). Each subagent gets its own `system_prompt`, `model_endpoint`, and a `tools:` list resolved against `SECTION_TOOL_REGISTRY` in `agent/agent_server/tools.py`.
- `assemble` — single `ChatDatabricks.ainvoke()` (no tools) that combines the three section markdowns into the final report. Uses the `report-assembler-agent` yaml entry.
- `save_report` — deterministic node (not an LLM tool call). Runs only when the inbound request includes `custom_inputs.save_location` starting with `/Volumes/`. Writes the markdown to that UC volume via `save_report_to_volume`.

Tools live in `agent/agent_server/tools.py` and are registered by name in `SECTION_TOOL_REGISTRY`: `search_earnings_docs`, `multi_query_search`, `list_available_periods`, `get_period_full_content`. `save_report_to_volume` is intentionally NOT in the registry — it's invoked only from the `save_report` graph node, never exposed to a subagent's LLM loop.

`@invoke()` / `@stream()` handlers (decorators from `mlflow.genai.agent_server`) wrap the graph as a `ResponsesAgent`-compatible endpoint. The stream handler currently runs the graph to completion and emits the final report as a single text output — true per-token streaming would require switching the assembler call to `.astream()`.

To add a new tool: define `@tool` in `tools.py`, add it to `SECTION_TOOL_REGISTRY`, then list its registry name under the relevant subagent's `tools:` in `agent_config.yaml`. To add a new section: add a yaml entry, register it in `_SECTION_TO_AGENT_NAME` / `_SECTION_ORDER` / `_SECTION_HEADERS` in `agent.py`.

## Notebook ↔ script duality

Every `.py` file under `doc_ingestion/` starts with `# Databricks notebook source` and uses `# COMMAND ----------` cell markers, so it works as a workspace notebook. The same files also run as standalone scripts (Databricks Connect, `databricks bundle run`) thanks to this idiom at the top of each:

```python
try:
    here = Path(__file__).resolve().parent
except NameError:
    notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    here = Path("/Workspace") / Path(notebook_path).parent.relative_to("/")
```

Preserve this pattern when adding new pipeline files — the rest of the pipeline assumes `here.parent / "ingestion_config.yaml"` resolves correctly in both modes.

## Compute requirements

- `ai_classify` / `ai_extract` v2 (notebook 01): serverless SQL warehouse, serverless compute, or DBR ≥ 16.4 shared. Will fail on a standard interactive cluster.
- `ai_parse_document` v2 (notebook 02): DBR ≥ 17.3 on serverless environment v3+ (or equivalent).
- `ai_query` + Pillow crop UDF (notebook 03): serverless. Pillow is in the default serverless Python env. The endpoint configured in `ingestion_config.yaml` (`databricks-claude-sonnet-4` by default) must be deployed and accessible.
- `ai_query` text-only (notebook 05): serverless. Same chat endpoint as 03, no image input. Cost scales with row count (~one chat call per page in `silver_document_elements_aggregated`).
- Agent runtime (`agent/`): the section + orchestration model serving endpoints declared in `agent_config.yaml` (and bound in `databricks.yml` under `apps.research_agent.resources`) must be deployed and grant `CAN_QUERY` to the app's service principal. The Vector Search index `gold_documents_index` and the source `gold_documents` table need explicit grants — see the comment block above `apps.research_agent` in `databricks.yml` for the one-time SQL grants required after first deploy.

## Auth

Two CLI profiles point at the same workspace (`https://fe-vm-vdm-classic-hkbucz.cloud.databricks.com`): `DEFAULT` and `fe-vm-vdm-classic-hkbucz`. Because both match, `databricks.yml` does not pin a profile and bare `databricks` commands fail with "multiple profiles matched". Always pass `--profile DEFAULT` (the working one) explicitly, or set `DATABRICKS_CONFIG_PROFILE=DEFAULT`. The MCP server in `.mcp.json` already pins `DEFAULT`.

## Common commands

```bash
# Local dev environment (Databricks Connect)
pip install -r requirements.txt   # databricks-connect>=16.4,<18, databricks-sdk, pyyaml

# Bundle workflow (always pass --profile DEFAULT)
databricks bundle validate --profile DEFAULT
databricks bundle deploy   --profile DEFAULT   # uploads to /Workspace/Users/<me>/.bundle/difficult-doc-qa/dev/files
databricks bundle summary  --profile DEFAULT
```

Bundle target `dev` is the default (`mode: development`). `databricks.yml` declares one MLflow experiment (`research_agent_experiment`) and two Databricks Apps (`doc_gallery_app`, `research_agent`); `bundle deploy` syncs files and (re)provisions those resources. To run a pipeline notebook, either open it in the workspace after deploy, or invoke it via Databricks Connect from a local venv (the `__file__`/`NameError` fallback handles both).

Some grants required by both apps (USE CATALOG, USE SCHEMA, Vector Search ACLs, table SELECT, optional volume read/write) are NOT bindable through the bundle's `databricks_app.resources` DSL and must be granted out-of-band to the app's service principal — the comment blocks above each `apps.*` entry in `databricks.yml` enumerate the exact SQL.
