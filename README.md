# difficult-doc-qa

A Databricks Asset Bundle that ingests earnings slide decks (PDF/PPTX) into a Vector Search index and serves a multi-agent research-report workflow on top of it. Three independent modules ship together in one bundle:

| Module | Path | What it does | Compute / runtime |
|---|---|---|---|
| **Ingestion** | `doc_ingestion/` | Medallion pipeline (bronze → silver → gold) over PDFs/PPTX. Auto Loader → `ai_parse_document` → `ai_query` figure captions → `ai_query` per-page summaries → Vector Search index. | Notebooks on serverless. |
| **Gallery app** | `doc_gallery_app/` | Streamlit Databricks App for browsing parsed pages side-by-side with their extracted elements (page image + element overlays + figure captions). | Databricks App. |
| **Research agent** | `agent/` | LangGraph research-report agent. Three parallel section subagents (Overview / Financial Performance / Devil's Advocate) + an assembler, served as a Databricks App via MLflow's `ResponsesAgent`. | Databricks App. |

The gallery app reads `gold_documents_gallery`. The research agent reads the Vector Search index `gold_documents_index` and (for period-drilling tools) the source `gold_documents` table. **Both apps depend on the ingestion pipeline having run at least once.**

Per-module config files (edit these rather than hardcoding):

- `doc_ingestion/ingestion_config.yaml` — target tables, catalog/schema/volume names, captioning endpoint + prompt, ticker allow-list.
- `agent/agent_config.yaml` — subagent prompts, model endpoints, per-subagent tool lists, retrieval columns.

See `CLAUDE.md` for deeper architectural notes (pipeline flow, agent graph shape, notebook ↔ script duality, checkpoint reprocessing).

---

## Prerequisites

- Databricks CLI installed and authenticated to the target workspace.
- A CLI profile that resolves to a single workspace. Two profiles point at the same host on this project (`DEFAULT` and `fe-vm-vdm-classic-hkbucz`), so bare `databricks` commands fail with "multiple profiles matched". **Always pass `--profile DEFAULT`** (or set `DATABRICKS_CONFIG_PROFILE=DEFAULT`).
- A serverless SQL warehouse the gallery app and agent can use. The default in `databricks.yml` is `var.sql_warehouse_id = 995bc1ffd2ff99a4` — override per environment.
- For local dev: `pip install -r requirements.txt` (Databricks Connect + SDK + pyyaml).

---

## Deploy via DABs (recommended)

The whole bundle deploys as one unit, but each module needs a different post-deploy action to actually become live.

```bash
# Validate the bundle.
databricks bundle validate --profile DEFAULT

# Sync source to /Workspace/Users/<me>/.bundle/difficult-doc-qa/dev/files/
# and provision declared resources (1 MLflow experiment + 2 Apps).
databricks bundle deploy --profile DEFAULT

databricks bundle summary --profile DEFAULT
```

After `bundle deploy`:

### Ingestion pipeline

`bundle deploy` only syncs the notebook files; **no Jobs are declared**, so nothing runs automatically. To execute the pipeline, run the notebooks in order from the synced workspace path:

```
/Workspace/Users/<me>/.bundle/difficult-doc-qa/dev/files/doc_ingestion/setup/load_files_to_volume
/Workspace/Users/<me>/.bundle/difficult-doc-qa/dev/files/doc_ingestion/pipeline/01_extract_metadata
…
/Workspace/Users/<me>/.bundle/difficult-doc-qa/dev/files/doc_ingestion/pipeline/06b_build_gallery
```

Each runs on serverless. Order matters — see `CLAUDE.md` ("Pipeline shape").

### Gallery app

```bash
# Provisions the Apps spec on first deploy. To deploy/redeploy the app code:
databricks bundle run doc_gallery_app --profile DEFAULT
```

First-deploy only — apply the catalog/schema USE grants the bundle DSL can't express (the comment block above `apps.doc_gallery_app` in `databricks.yml` lists them):

```bash
SP_ID=$(databricks apps get doc-gallery-app --profile DEFAULT --output json | jq -r .service_principal_client_id)

databricks sql --profile DEFAULT --warehouse-id <warehouse-id> --query "
  GRANT USE CATALOG ON CATALOG doan TO \`$SP_ID\`;
  GRANT USE SCHEMA  ON SCHEMA  doan.difficult_doc_qa TO \`$SP_ID\`;
"
```

### Research agent

```bash
databricks bundle run research_agent --profile DEFAULT
```

First-deploy only — the agent needs more out-of-band grants than the gallery (catalog/schema USE, Vector Search endpoint + index ACLs, source table SELECT, optional save volume). See the comment block above `apps.research_agent` in `databricks.yml` for the full SQL.

```bash
SP_ID=$(databricks apps get research-agent --profile DEFAULT --output json | jq -r .service_principal_client_id)
# then run the GRANTs from databricks.yml against your SQL warehouse
```

Test the deployed agent endpoint via `agent/scripts/test_agent.py` (open the synced copy in the workspace, attach serverless, Run All) or hit `/invocations` directly with the URL from `databricks apps get research-agent`.

---

## Deploy manually in the workspace (no bundle)

Useful when you don't want bundle-managed lifecycle (e.g. iterating on a single notebook, or shipping one module to a workspace that doesn't have the bundle deployed).

### Ingestion pipeline (manual)

1. **Get the notebooks into the workspace.** Either drag-and-drop the `doc_ingestion/` folder via the UI (Workspace → ⋯ → Import), or sync from CLI:
   ```bash
   databricks workspace import-dir doc_ingestion /Workspace/Users/<me>/doc_ingestion --profile DEFAULT
   ```
2. **Open `doc_ingestion/setup/load_files_to_volume`** and run it once to upload sample PDFs/PPTX from `setup/data/` into the configured volume. Idempotent.
3. **Run `pipeline/01` … `pipeline/06b` in order**, each on serverless. Notebooks 01 and 02 require recent DBR (`ai_classify` v2 needs DBR ≥ 16.4 shared / serverless; `ai_parse_document` v2 needs DBR ≥ 17.3 / serverless env v3+). See `CLAUDE.md` ("Compute requirements").
4. **To force reprocessing**, bump the relevant `*_checkpoint_prefix` value in `ingestion_config.yaml` AND drop the affected output rows — Auto Loader is path-keyed, so without bumping the prefix `Trigger.AvailableNow` treats the source as already consumed.

### Gallery app (manual)

```bash
# 1. Sync the app source into the workspace.
databricks sync doc_gallery_app /Workspace/Users/<me>/doc-gallery-app --profile DEFAULT

# 2. Create the app (one-time).
databricks apps create doc-gallery-app --profile DEFAULT

# 3. Deploy the synced source.
databricks apps deploy doc-gallery-app \
  --source-code-path /Workspace/Users/<me>/doc-gallery-app \
  --profile DEFAULT
```

The bundle's resource bindings (SQL warehouse, gallery table, page-images volume) won't be applied without DABs. Either bind them manually in the Apps UI (App settings → Resources), or grant them directly to the app's service principal:

```sql
GRANT USE CATALOG ON CATALOG doan TO `<sp_client_id>`;
GRANT USE SCHEMA  ON SCHEMA  doan.difficult_doc_qa TO `<sp_client_id>`;
GRANT SELECT      ON TABLE   doan.difficult_doc_qa.gold_documents_gallery TO `<sp_client_id>`;
GRANT READ VOLUME ON VOLUME  doan.difficult_doc_qa.ai_parse_doc_images TO `<sp_client_id>`;
```

The env vars in `doc_gallery_app/app.yaml` (`DATABRICKS_WAREHOUSE_ID`, `GALLERY_TABLE`) need a real warehouse id — without the bundle's `valueFrom: sql-warehouse` wiring you'll need to either set `DATABRICKS_WAREHOUSE_ID` to a literal id in `app.yaml` or bind the warehouse resource manually in the UI.

### Research agent (manual)

```bash
# 1. Sync source.
databricks sync agent /Workspace/Users/<me>/research-agent --profile DEFAULT

# 2. Create the app.
databricks apps create research-agent --profile DEFAULT

# 3. Deploy.
databricks apps deploy research-agent \
  --source-code-path /Workspace/Users/<me>/research-agent \
  --profile DEFAULT
```

`agent/app.yaml` declares the env vars and resource bindings the runtime expects (`MLFLOW_EXPERIMENT_ID`, `DATABRICKS_WAREHOUSE_ID`). Without DABs, bind these in the Apps UI: Experiment, two serving endpoints (`databricks-claude-sonnet-4-6` + `databricks-claude-haiku-4-5` by default — see `var.research_agent_*_endpoint` in `databricks.yml`), and a SQL warehouse.

Then apply the same out-of-band SQL grants the bundle path requires (catalog/schema USE, Vector Search ACLs, `gold_documents` SELECT, optional `research_reports` volume) — the comment block above `apps.research_agent` in `databricks.yml` is the canonical list.

For local iteration without redeploying, the in-process driver at `agent/scripts/test_agent.py` runs the LangGraph agent directly via Databricks Connect — much faster than `apps deploy` cycles.

---

## Common pitfalls

- **"multiple profiles matched"** — pass `--profile DEFAULT` explicitly or export `DATABRICKS_CONFIG_PROFILE=DEFAULT`.
- **Pipeline reprocessing did nothing** — you bumped the data but not the checkpoint prefix (or vice versa). Both are required.
- **App deploys but 500s on first request** — out-of-band SQL grants weren't applied. Check the app logs; the failing grant is usually named in the stack trace.
- **`save_report` returns `ERROR: ... does not have READ VOLUME`** — the SDK's `files.upload` reads metadata before writing, so `WRITE VOLUME` alone isn't enough. Grant both `READ VOLUME` and `WRITE VOLUME`.
