# Databricks notebook source
# MAGIC %md
# MAGIC # Load source documents into a Unity Catalog volume
# MAGIC
# MAGIC Reads `doc_ingestion/ingestion_config.yaml`, ensures the target
# MAGIC catalog/schema/volume exist, and uploads every file from
# MAGIC `doc_ingestion/setup/data/` into the volume so downstream
# MAGIC `ai_parse_document` jobs can read them.
# MAGIC
# MAGIC Works both as a workspace notebook and via Databricks Connect from a
# MAGIC local venv — uploads go through the SDK Files API rather than the
# MAGIC `/Volumes` FUSE mount (which only exists on Databricks compute).

# COMMAND ----------

from pathlib import Path

import yaml
from databricks.sdk import WorkspaceClient

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve paths and load config
# MAGIC
# MAGIC `__file__` is defined when this runs as a script (Databricks Connect,
# MAGIC `databricks bundle run`, etc.). In the workspace notebook UI it isn't,
# MAGIC so fall back to the notebook context path.

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

config_path = here.parent / "ingestion_config.yaml"
data_dir = here / "data"

with open(config_path) as f:
    config = yaml.safe_load(f)["databricks_config"]

catalog = config["catalog"]
schema = config["schema"]
volume = config["volume_name"]
volume_path = f"/Volumes/{catalog}/{schema}/{volume}"

print(f"config:   {config_path}")
print(f"data dir: {data_dir}")
print(f"target:   {volume_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create catalog, schema, and volume

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upload files to the volume
# MAGIC
# MAGIC Skips hidden files (e.g. `.DS_Store`) and anything already present at
# MAGIC the destination with a matching size, so re-runs are idempotent.

# COMMAND ----------

if not data_dir.exists():
    raise FileNotFoundError(f"Source data directory not found: {data_dir}")

w = WorkspaceClient()

existing = {}
try:
    for entry in w.files.list_directory_contents(volume_path):
        existing[entry.name] = entry.file_size
except Exception:
    pass

source_files = sorted(p for p in data_dir.iterdir() if p.is_file() and not p.name.startswith("."))
print(f"Found {len(source_files)} source file(s)")

uploaded, skipped = 0, 0
for src in source_files:
    size = src.stat().st_size
    if existing.get(src.name) == size:
        skipped += 1
        continue
    with open(src, "rb") as f:
        w.files.upload(f"{volume_path}/{src.name}", f, overwrite=True)
    uploaded += 1
    print(f"  uploaded  {src.name}  ({size / 1_000_000:.1f} MB)")

print(f"\nDone. Uploaded {uploaded}, skipped {skipped}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

for entry in w.files.list_directory_contents(volume_path):
    print(f"{entry.file_size:>12}  {entry.name}")
