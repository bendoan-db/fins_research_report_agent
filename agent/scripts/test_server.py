# Databricks notebook source
# MAGIC %md
# MAGIC # Smoke-test the deployed research-agent app
# MAGIC
# MAGIC POSTs an inference request to the deployed Databricks App's
# MAGIC `/invocations` endpoint. Unlike `test_agent.py` (which loads the
# MAGIC LangGraph in-process), this exercises the full HTTPS round-trip
# MAGIC against the production-style deployment.
# MAGIC
# MAGIC `save_location` is forwarded as `custom_inputs.save_location`; the
# MAGIC agent routes through its `save_report` node to write the final
# MAGIC markdown to `{save_location}/{ticker}_{utc_iso}.md`. Set to an
# MAGIC empty string to skip the save and only receive the report inline.
# MAGIC
# MAGIC **Run from anywhere** — locally as a script
# MAGIC (`DATABRICKS_CONFIG_PROFILE=DEFAULT python scripts/test_server.py`)
# MAGIC or as a workspace notebook. Auth is resolved by
# MAGIC `databricks.sdk.WorkspaceClient` (PAT, OAuth U2M, M2M).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install dependencies
# MAGIC
# MAGIC `requests` for the HTTP POST; `databricks-sdk` for auth header
# MAGIC resolution. Both ship in the default serverless image but pin
# MAGIC for clarity. The `%pip` / `%restart_python` magics are no-ops
# MAGIC when run as a standalone Python script.

# COMMAND ----------

# MAGIC %pip install --quiet "requests>=2.31" "databricks-sdk>=0.30"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters
# MAGIC
# MAGIC Edit these for a different ticker, deployment, or save target.
# MAGIC The default `app_url` is the URL printed by `databricks bundle run
# MAGIC research_agent --profile DEFAULT` after a successful deploy.

# COMMAND ----------

question = "Research report for NVDA"
save_location = "/Volumes/doan/difficult_doc_qa/research_reports"
app_url = "https://research-agent-2309167578215964.aws.databricksapps.com"
timeout_seconds = 600

print(f"app_url:       {app_url}")
print(f"question:      {question}")
print(f"save_location: {save_location or '(none)'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Send the request
# MAGIC
# MAGIC Auth resolves to whichever method the SDK can find. The bearer
# MAGIC token from `cfg.authenticate()` is forwarded directly to the
# MAGIC app's HTTPS endpoint.

# COMMAND ----------

import time

import requests
from databricks.sdk import WorkspaceClient

cfg = WorkspaceClient().config
auth_headers = cfg.authenticate()

body: dict = {"input": [{"role": "user", "content": question}]}
if save_location:
    body["custom_inputs"] = {"save_location": save_location}

start = time.time()
response = requests.post(
    f"{app_url.rstrip('/')}/invocations",
    headers={**auth_headers, "Content-Type": "application/json"},
    json=body,
    timeout=timeout_seconds,
)
elapsed = time.time() - start

print(f"--- HTTP {response.status_code} in {elapsed:.1f}s ---")
if response.status_code != 200:
    # Surface the server's error body and stop — easier to read than a
    # Python traceback when the failure is upstream (auth, UC grants,
    # tool runtime errors).
    print(response.text)
    raise SystemExit(1)

payload = response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract the report
# MAGIC
# MAGIC ResponsesAgent payload shape:
# MAGIC `{"output": [{"type": "message", "content": [{"type": "output_text", "text": "..."}]}]}`.

# COMMAND ----------

import json

chunks: list[str] = []
for item in payload.get("output") or []:
    if item.get("type") != "message":
        continue
    for content in item.get("content") or []:
        if content.get("type") == "output_text":
            chunks.append(content.get("text") or "")
answer = "\n".join(chunks)

print(f"--- final-report length: {len(answer):,} chars ---")
if not answer:
    print("(no output_text item found — raw payload:)")
    print(json.dumps(payload, indent=2)[:2000])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rendered report
# MAGIC
# MAGIC The cell below renders the assembled markdown inline when run in
# MAGIC a notebook. Locally (CLI) it prints the first 1500 chars instead.

# COMMAND ----------

try:
    # Notebook path: render markdown inline.
    from IPython.display import Markdown  # type: ignore

    Markdown(answer)
except ImportError:
    # CLI path: just print a preview.
    print("--- first 1500 chars of report ---")
    print(answer[:1500])
