# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Caption figures via multimodal `ai_query`
# MAGIC
# MAGIC For every row in `silver_document_elements` where `type = 'figure'`,
# MAGIC crop the corresponding page image using the bbox coordinates and pass
# MAGIC the cropped bytes to `ai_query` with the captioning prompt configured
# MAGIC in `ingestion_config.yaml`. Results land in
# MAGIC `<catalog>.<schema>.<figure_captions_table>` (default
# MAGIC `silver_figure_captions`), joinable back to the elements table on
# MAGIC `(path, element_id)`.
# MAGIC
# MAGIC Pipeline shape:
# MAGIC
# MAGIC 1. Stream `silver_document_elements` filtered to `type='figure'`.
# MAGIC 2. Pull `bbox[0].coord` and `bbox[0].page_id` out of the VARIANT.
# MAGIC 3. Static-join with `silver_document_pages` on `(path, page_id)` to
# MAGIC    fetch the page `image_uri`.
# MAGIC 4. Python UDF reads the page image from the volume FUSE mount, crops
# MAGIC    with PIL, returns PNG bytes (BINARY).
# MAGIC 5. `ai_query(<endpoint>, <prompt>, files => cropped)` returns the
# MAGIC    caption STRING.
# MAGIC 6. Write to the captions table with a per-table checkpoint.
# MAGIC
# MAGIC **Multi-region figures**: only `bbox[0]` is used. Sample data shows
# MAGIC length-1 bboxes in practice; if a real multi-region figure shows up,
# MAGIC we'd need to revisit (e.g. crop a union or caption each region).
# MAGIC
# MAGIC **Compute**: serverless. The UDF needs Pillow, which is part of the
# MAGIC default serverless Python env. `ai_query` needs the configured vision
# MAGIC endpoint to be deployed and accessible.

# COMMAND ----------

import io
from pathlib import Path

import yaml
from pyspark.sql.types import BinaryType

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
volume = db["volume_name"]

dp = config["ingestion_pipeline"]["document_parsing_step"]
fc = config["ingestion_pipeline"]["figure_captioning_step"]

elements_table = dp["parsed_document_elements_table"]
pages_table = dp["parsed_document_pages_table"]
captions_table = fc["figure_captions_table"]
caption_endpoint = fc["caption_endpoint"]
caption_prompt = " ".join(fc["captioning_prompt"].split())

source_volume_path = f"/Volumes/{catalog}/{schema}/{volume}"
elements_table_fqn = f"{catalog}.{schema}.{elements_table}"
pages_table_fqn = f"{catalog}.{schema}.{pages_table}"
captions_table_fqn = f"{catalog}.{schema}.{captions_table}"
captions_checkpoint = f"{source_volume_path}/{fc['caption_checkpoint_prefix']}/{captions_table}"

print(f"elements:    {elements_table_fqn}")
print(f"pages:       {pages_table_fqn}")
print(f"captions:    {captions_table_fqn}")
print(f"endpoint:    {caption_endpoint}")
print(f"prompt:      {caption_prompt[:100]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Crop UDF
# MAGIC
# MAGIC Reads the page image directly from the UC volume's FUSE mount, crops
# MAGIC `(x1, y1, x2, y2)` with PIL, and returns PNG bytes. Errors swallowed
# MAGIC and reported as `NULL` so a single bad image doesn't kill the stream.

# COMMAND ----------

_MAX_SIDE = 1568  # Claude vision's native max — resizing here also keeps payload < 5 MB.


def crop_image_to_png(image_uri, x1, y1, x2, y2):
    if image_uri is None or x1 is None or y1 is None or x2 is None or y2 is None:
        return None
    try:
        from PIL import Image

        with Image.open(image_uri) as img:
            cropped = img.crop((int(x1), int(y1), int(x2), int(y2))).convert("RGB")
            cropped.thumbnail((_MAX_SIDE, _MAX_SIDE), Image.LANCZOS)
            buf = io.BytesIO()
            cropped.save(buf, format="PNG", optimize=True)
            return buf.getvalue()
    except Exception:
        return None


spark.udf.register("crop_image_to_png", crop_image_to_png, BinaryType())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build the caption stream

# COMMAND ----------

# Static lookup: page_id → image_uri for each document.
pages_static = (
    spark.read.table(pages_table_fqn)
    .selectExpr("path AS p_path", "page_id AS p_page_id", "image_uri")
)

# Streaming source: figure elements with bbox fields hoisted out of VARIANT.
figure_elements = (
    spark.readStream.table(elements_table_fqn)
    .where("type = 'figure'")
    .selectExpr(
        "path",
        "filename",
        "ticker",
        "quarter",
        "year",
        "element_id",
        "CAST(variant_get(bbox, '$[0].page_id', 'INT') AS INT) AS page_id",
        "CAST(variant_get(bbox, '$[0].coord[0]', 'INT') AS INT) AS x1",
        "CAST(variant_get(bbox, '$[0].coord[1]', 'INT') AS INT) AS y1",
        "CAST(variant_get(bbox, '$[0].coord[2]', 'INT') AS INT) AS x2",
        "CAST(variant_get(bbox, '$[0].coord[3]', 'INT') AS INT) AS y2",
        "bbox",
    )
)

joined = (
    figure_elements.join(
        pages_static,
        (figure_elements.path == pages_static.p_path)
        & (figure_elements.page_id == pages_static.p_page_id),
        "left",
    )
    .drop("p_path", "p_page_id")
)

# Escape single quotes in the prompt so it can be embedded in a SQL string literal.
prompt_sql = caption_prompt.replace("'", "''")

caption_stream = joined.selectExpr(
    "path",
    "filename",
    "ticker",
    "quarter",
    "year",
    "element_id",
    "page_id",
    "image_uri AS source_image_uri",
    "bbox",
    f"ai_query("
    f"  '{caption_endpoint}',"
    f"  '{prompt_sql}',"
    f"  files => crop_image_to_png(image_uri, x1, y1, x2, y2)"
    f") AS caption",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write captions

# COMMAND ----------

caption_query = (
    caption_stream.writeStream.option("checkpointLocation", captions_checkpoint)
    .trigger(availableNow=True)
    .toTable(captions_table_fqn)
)
caption_query.awaitTermination()
caption_input = sum(p.get("numInputRows", 0) for p in caption_query.recentProgress)
print(f"Captioned {caption_input} new figure(s).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

display(
    spark.sql(
        f"""
        SELECT
          ticker,
          COUNT(*)                                     AS n_figures,
          SUM(CASE WHEN caption IS NOT NULL THEN 1 ELSE 0 END) AS n_captioned,
          SUM(CASE WHEN caption IS NULL     THEN 1 ELSE 0 END) AS n_null
        FROM {captions_table_fqn}
        GROUP BY ticker
        ORDER BY ticker
        """
    )
)

# COMMAND ----------

display(
    spark.sql(
        f"""
        SELECT filename, element_id, page_id, LEFT(caption, 220) AS caption_preview
        FROM {captions_table_fqn}
        WHERE caption IS NOT NULL
        ORDER BY filename, element_id
        LIMIT 10
        """
    )
)
