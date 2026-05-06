# Databricks notebook source
# MAGIC %md
# MAGIC # Test the research-report agent
# MAGIC
# MAGIC Smoke-tests the agent against a single user question and renders the
# MAGIC final markdown report inline. Loads the agent in-process — no FastAPI
# MAGIC server, no `/invocations` round-trip — so it's the fastest way to
# MAGIC iterate on prompts and tools while watching MLflow traces.
# MAGIC
# MAGIC **Run from the bundle workspace path** (after `databricks bundle
# MAGIC deploy --profile DEFAULT`):
# MAGIC `/Workspace/Users/<me>/.bundle/difficult-doc-qa/dev/files/agent/scripts/test_agent`
# MAGIC
# MAGIC **Compute**: serverless. Auth + endpoint discovery are handled by the
# MAGIC Databricks runtime; the helper in `agent_server.tools` already extracts
# MAGIC a bearer token from whichever auth method the SDK resolves.
# MAGIC
# MAGIC **Prerequisite**: the Vector Search index `gold_documents_index` must
# MAGIC exist and be `ONLINE` on the `financebench` endpoint (see
# MAGIC `doc_ingestion/pipeline/06a_build_index.py`).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install agent dependencies
# MAGIC
# MAGIC The agent's runtime deps aren't part of the default serverless image.
# MAGIC Pinning major versions to match `agent/pyproject.toml`.

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   "databricks-langchain>=0.17.0" \
# MAGIC   "databricks-agents>=1.9.3" \
# MAGIC   "langgraph>=1.1.0" \
# MAGIC   "langchain>=1.0.0" \
# MAGIC   "mlflow>=3.10.0" \
# MAGIC   pyyaml

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve paths and put `agent_server` on `sys.path`

# COMMAND ----------

import sys
from pathlib import Path

try:
    here = Path(__file__).resolve().parent
except NameError:
    notebook_path = (
        dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .notebookPath()
        .get()
    )
    here = Path("/Workspace") / Path(notebook_path).parent.relative_to("/")

# scripts/ is a sibling of agent_server/. Adding the parent makes
# `from agent_server.agent import init_agent` resolve cleanly.
sys.path.insert(0, str(here.parent))

print(f"agent dir: {here.parent}")
print(f"sys.path[0]: {sys.path[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question + optional save location
# MAGIC
# MAGIC `question` — the user prompt sent to the agent.
# MAGIC
# MAGIC `save_location` — optional. When set to a `/Volumes/...` path, the
# MAGIC agent appends a `save_report` step that writes the final markdown to
# MAGIC `{save_location}/{ticker}_{utc_iso}.md`. Leave as `None` (or
# MAGIC empty string) to skip the save and behave as before. The deployed
# MAGIC SP needs `WRITE VOLUME` on whichever volume you point at — see the
# MAGIC inline grant block in `databricks.yml`.

# COMMAND ----------

question = "Research report for NVDA"
save_location = "/Volumes/doan/difficult_doc_qa/research_reports"
print(f">>> {question}")
if save_location:
    print(f"save_location: {save_location}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bind the MLflow experiment from `agent_config.yaml`
# MAGIC
# MAGIC `agent_server.agent` already calls `mlflow.set_experiment(...)` at
# MAGIC import time using the path declared in
# MAGIC `agent_config.yaml:agent_config.mlflow_experiment_name`. Doing it
# MAGIC again here makes the binding visible in the notebook output and lets
# MAGIC you click straight through to the MLflow experiment UI from the
# MAGIC printed link.

# COMMAND ----------

import mlflow

from agent_server.config import load_agent_config

# Pin tracking URI to databricks so workspace experiment paths (`/Workspace/...`)
# resolve correctly regardless of how the MLFLOW_TRACKING_URI env var is set.
mlflow.set_tracking_uri("databricks")

experiment_name = load_agent_config()["agent_config"]["mlflow_experiment_name"]
experiment = mlflow.set_experiment(experiment_name)
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
print(f"MLflow experiment:   {experiment_name}")
print(f"          id:        {experiment.experiment_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Invoke the agent
# MAGIC
# MAGIC Builds the LangGraph `StateGraph` (extract_ticker → fan-out to three
# MAGIC parallel section writers via `Send` → assemble → END) and runs it
# MAGIC against the question. All graph-node / subagent / tool spans land in
# MAGIC the experiment bound above; `mlflow.langchain.autolog()` is enabled
# MAGIC by `agent_server.agent` on import.

# COMMAND ----------

import asyncio
import time

from agent_server.agent import init_agent

graph = init_agent()

start = time.time()
initial_state = {"user_message": question, "sections": []}
if save_location:
    initial_state["save_location"] = save_location
result = asyncio.run(graph.ainvoke(initial_state))
elapsed = time.time() - start

answer = result.get("final_report", "") or ""

print(f"--- elapsed: {elapsed:.1f}s ---")
print(f"--- final-report length: {len(answer):,} chars ---")
if saved_path := result.get("saved_path"):
    print(f"--- saved_path: {saved_path} ---")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rendered report
# MAGIC
# MAGIC The cell below renders the agent's markdown output inline. The raw
# MAGIC text is still in the `answer` variable if you want to copy it into a
# MAGIC `.md` file.

# COMMAND ----------

from IPython.display import Markdown

Markdown(answer)
