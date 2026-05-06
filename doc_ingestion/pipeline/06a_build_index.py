# Databricks notebook source
# MAGIC %md
# MAGIC # 06a — Build (or sync) the Databricks Vector Search index over `gold_documents`
# MAGIC
# MAGIC Idempotent setup step that creates the Delta-sync index the
# MAGIC research-report agent retrieves from. If the index already exists, the
# MAGIC notebook just triggers a sync; otherwise it issues `create_delta_sync_index`
# MAGIC with managed embeddings.
# MAGIC
# MAGIC Configuration source: `ingestion_config.yaml` (`vector_index_step` +
# MAGIC `summarization_step.gold_documents_table`).
# MAGIC
# MAGIC | Setting             | Value                                              |
# MAGIC | ------------------- | -------------------------------------------------- |
# MAGIC | Endpoint            | `vector_index_step.vector_search_endpoint`         |
# MAGIC | Index name          | `vector_index_step.vector_search_index`            |
# MAGIC | Source table        | `summarization_step.gold_documents_table`          |
# MAGIC | Primary key         | `uuid` (added in step 04 / propagated in step 05)  |
# MAGIC | Embedding source    | `content_to_embed`                                 |
# MAGIC | Embedding endpoint  | `vector_index_step.embedding_model`                |
# MAGIC | Sync mode           | `TRIGGERED` Delta-sync                             |
# MAGIC
# MAGIC **Prerequisite**: the Vector Search endpoint must exist (created out-of-band
# MAGIC by a Vector Search admin or via `databricks vector-search-endpoints
# MAGIC create-endpoint`). The notebook fails fast with a clear message if not.
# MAGIC
# MAGIC **Run order**: must run after step 05 so `gold_documents` is populated
# MAGIC with `uuid` + `content_to_embed`. Run before any agent that queries the
# MAGIC index.

# COMMAND ----------

import sys
from pathlib import Path

import yaml
from databricks.vector_search.client import VectorSearchClient

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve paths and load config

# COMMAND ----------

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

with open(here.parent / "ingestion_config.yaml") as f:
    config = yaml.safe_load(f)

db = config["databricks_config"]
catalog = db["catalog"]
schema = db["schema"]

ss = config["ingestion_pipeline"]["summarization_step"]
vi = config["ingestion_pipeline"]["vector_index_step"]

source_table_full_name = f"{catalog}.{schema}.{ss['gold_documents_table']}"
endpoint_name = vi["vector_search_endpoint"]
index_full_name = f"{catalog}.{schema}.{vi['vector_search_index']}"
embedding_model = vi["embedding_model"]

print(f"endpoint:      {endpoint_name}")
print(f"index:         {index_full_name}")
print(f"source table:  {source_table_full_name}")
print(f"embedding:     {embedding_model}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify the endpoint exists

# COMMAND ----------

client = VectorSearchClient(disable_notice=True)

try:
    client.get_endpoint(endpoint_name)
except Exception:
    print(
        f"ERROR: endpoint '{endpoint_name}' not found. Create it first via the "
        f"Databricks UI (Compute → Vector Search) or:\n"
        f"  databricks vector-search-endpoints create-endpoint --name {endpoint_name} --endpoint-type STANDARD",
        file=sys.stderr,
    )
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create or sync the index

# COMMAND ----------

try:
    existing = client.get_index(endpoint_name=endpoint_name, index_name=index_full_name)
    print("index exists; triggering sync …")
    existing.sync()
    print("sync triggered")
except Exception:
    print("creating index (TRIGGERED Delta-sync, managed embeddings) …")
    client.create_delta_sync_index(
        endpoint_name=endpoint_name,
        index_name=index_full_name,
        source_table_name=source_table_full_name,
        pipeline_type="TRIGGERED",
        primary_key="uuid",
        embedding_source_column="content_to_embed",
        embedding_model_endpoint_name=embedding_model,
        # Explicit list — relying on the default subset has caused INT
        # columns (page_id, n_elements) to be silently dropped before.
        columns_to_sync=[
            "uuid",
            "path",
            "filename",
            "ticker",
            "quarter",
            "year",
            "page_id",
            "n_elements",
            "image_uri",
            "content_to_embed",
        ],
    )
    print("index creation submitted — initial sync will run automatically.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

idx = client.get_index(endpoint_name=endpoint_name, index_name=index_full_name)
desc = idx.describe()
print(f"state:        {desc.get('status', {}).get('detailed_state', 'UNKNOWN')}")
print(f"ready:        {desc.get('status', {}).get('ready', False)}")
print(f"row count:    {desc.get('status', {}).get('indexed_row_count', '?')}")
print(
    "track status with:\n"
    f"  databricks vector-search-indexes get {index_full_name}"
)
