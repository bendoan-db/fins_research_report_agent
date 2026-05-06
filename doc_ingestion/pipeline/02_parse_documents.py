# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Parse documents and project into silver pages/elements tables
# MAGIC
# MAGIC Reads the staged binary contents from the metadata table written by
# MAGIC `01_extract_metadata.py`, runs `ai_parse_document` v2 with image
# MAGIC extraction + figure descriptions, and writes three Delta tables:
# MAGIC
# MAGIC 1. `<catalog>.<schema>.<parsed_document_table>` — one row per source
# MAGIC    document, with the full parsed VARIANT preserved.
# MAGIC 2. `<catalog>.<schema>.<parsed_document_pages_table>` — one row per
# MAGIC    page, with `image_uri` pointing at the image written under
# MAGIC    `imageOutputPath`.
# MAGIC 3. `<catalog>.<schema>.<parsed_document_elements_table>` — one row per
# MAGIC    extracted element (text/table/figure/title/...), with figure
# MAGIC    descriptions inlined.
# MAGIC
# MAGIC All three stages stream from a Delta source with `Trigger.AvailableNow`
# MAGIC and a per-table checkpoint, so re-runs only process new source rows.
# MAGIC Each stream's checkpoint sits at
# MAGIC `<volume>/<stage_checkpoint_prefix>/<table_name>`, where the prefix
# MAGIC comes from `ingestion_pipeline.document_parsing_step.*_checkpoint_prefix`.
# MAGIC
# MAGIC **Compute:** `ai_parse_document` v2 requires DBR ≥ 17.3 on serverless
# MAGIC environment v3+ (or equivalent). Will not run on a standard interactive
# MAGIC cluster.
# MAGIC
# MAGIC **Re-parsing a document:** bump the prefix value (e.g. `_checkpoints/v1`
# MAGIC → `_checkpoints/v2`) for the stages you want to redo, and drop the
# MAGIC affected rows from the corresponding tables — otherwise the silver
# MAGIC tables will accumulate duplicate pages/elements.

# COMMAND ----------

from pathlib import Path

import yaml

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
ai_parse_doc_volume = db["ai_parse_doc_volume"]

me = config["ingestion_pipeline"]["metadata_extraction_step"]
dp = config["ingestion_pipeline"]["document_parsing_step"]

metadata_table = me["document_metadata_table"]
parsed_table = dp["parsed_document_table_raw_elements_table"]
pages_table = dp["parsed_document_pages_table"]
elements_table = dp["parsed_document_elements_table"]

source_volume_path = f"/Volumes/{catalog}/{schema}/{volume}"
image_output_path = f"/Volumes/{catalog}/{schema}/{ai_parse_doc_volume}"

metadata_table_fqn = f"{catalog}.{schema}.{metadata_table}"
parsed_table_fqn = f"{catalog}.{schema}.{parsed_table}"
pages_table_fqn = f"{catalog}.{schema}.{pages_table}"
elements_table_fqn = f"{catalog}.{schema}.{elements_table}"

parsed_checkpoint = f"{source_volume_path}/{dp['parsed_document_table_raw_elements_checkpoint_prefix']}/{parsed_table}"
pages_checkpoint = f"{source_volume_path}/{dp['parsed_document_table_pages_checkpoint_prefix']}/{pages_table}"
elements_checkpoint = f"{source_volume_path}/{dp['parsed_document_table_elements_checkpoint_prefix']}/{elements_table}"

print(f"source:           {metadata_table_fqn}")
print(f"parsed target:    {parsed_table_fqn}")
print(f"pages target:     {pages_table_fqn}")
print(f"elements target:  {elements_table_fqn}")
print(f"image output:     {image_output_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure the image-output volume exists

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{ai_parse_doc_volume}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 1 — Parse: metadata table → parsed table
# MAGIC
# MAGIC Each source row carries `content` (BINARY). `ai_parse_document` v2 with
# MAGIC `imageOutputPath` writes per-page images into the configured volume and
# MAGIC returns a VARIANT containing the structured parse. Raw `content` is
# MAGIC dropped from the output — we don't need to keep two copies.

# COMMAND ----------

parse_options_sql = (
    "map("
    "'version', '2.0', "
    f"'imageOutputPath', '{image_output_path}', "
    "'descriptionElementTypes', 'figure'"
    ")"
)

parse_stream = (
    spark.readStream.table(metadata_table_fqn)
    .selectExpr(
        "path",
        "filename",
        "ticker",
        "quarter",
        "year",
        "modification_time",
        "size_bytes",
        f"ai_parse_document(content, {parse_options_sql}) AS parsed",
    )
)

parse_query = (
    parse_stream.writeStream.option("checkpointLocation", parsed_checkpoint)
    .trigger(availableNow=True)
    .toTable(parsed_table_fqn)
)
parse_query.awaitTermination()
parse_count = sum(p.get("numInputRows", 0) for p in parse_query.recentProgress)
print(f"Stage 1 (parse): {parse_count} new document(s) parsed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 2 — Pages: parsed table → silver pages
# MAGIC
# MAGIC Casts `parsed:document.pages` into a typed array of
# MAGIC `STRUCT<id:INT, image_uri:STRING>` and explodes it. One output row per
# MAGIC page, keyed by `(path, page_id)`.

# COMMAND ----------

pages_stream = (
    spark.readStream.table(parsed_table_fqn)
    .selectExpr(
        "path",
        "filename",
        "ticker",
        "quarter",
        "year",
        "explode("
        "  variant_get(parsed, '$.document.pages',"
        "              'ARRAY<STRUCT<id:INT, image_uri:STRING>>')"
        ") AS page",
    )
    .selectExpr(
        "path",
        "filename",
        "ticker",
        "quarter",
        "year",
        "page.id        AS page_id",
        "page.image_uri AS image_uri",
    )
)

pages_query = (
    pages_stream.writeStream.option("checkpointLocation", pages_checkpoint)
    .trigger(availableNow=True)
    .toTable(pages_table_fqn)
)
pages_query.awaitTermination()
pages_input = sum(p.get("numInputRows", 0) for p in pages_query.recentProgress)
print(f"Stage 2 (pages): exploded {pages_input} parsed document(s) into pages.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 3 — Elements: parsed table → silver elements
# MAGIC
# MAGIC Casts `parsed:document.elements` into a typed array and explodes it.
# MAGIC `bbox` is left as VARIANT — schema may evolve and downstream queries
# MAGIC can extract whatever they need with `variant_get`. `page_id` is hoisted
# MAGIC out of `bbox[0].page_id` for joins with `silver_document_pages` without
# MAGIC re-parsing the VARIANT downstream. Multi-region elements (rare in
# MAGIC practice) keep their full bbox array; only the first region's page is
# MAGIC surfaced as the canonical `page_id`.

# COMMAND ----------

elements_stream = (
    spark.readStream.table(parsed_table_fqn)
    .selectExpr(
        "path",
        "filename",
        "ticker",
        "quarter",
        "year",
        "explode("
        "  variant_get(parsed, '$.document.elements',"
        "              'ARRAY<STRUCT<id:INT, type:STRING, content:STRING,"
        "                            confidence:DOUBLE, description:STRING,"
        "                            bbox:VARIANT>>')"
        ") AS el",
    )
    .selectExpr(
        "path",
        "filename",
        "ticker",
        "quarter",
        "year",
        "el.id          AS element_id",
        "el.type        AS type",
        "el.content     AS content",
        "el.confidence  AS confidence",
        "el.description AS description",
        "CAST(variant_get(el.bbox, '$[0].page_id', 'INT') AS INT) AS page_id",
        "el.bbox        AS bbox",
    )
)

elements_query = (
    elements_stream.writeStream.option("checkpointLocation", elements_checkpoint)
    .trigger(availableNow=True)
    .toTable(elements_table_fqn)
)
elements_query.awaitTermination()
elements_input = sum(p.get("numInputRows", 0) for p in elements_query.recentProgress)
print(f"Stage 3 (elements): exploded {elements_input} parsed document(s) into elements.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

display(
    spark.sql(
        f"""
        SELECT 'parsed'   AS table_name, COUNT(*) AS n FROM {parsed_table_fqn}
        UNION ALL
        SELECT 'pages',    COUNT(*) FROM {pages_table_fqn}
        UNION ALL
        SELECT 'elements', COUNT(*) FROM {elements_table_fqn}
        """
    )
)

# COMMAND ----------

display(
    spark.sql(
        f"""
        SELECT type, COUNT(*) AS n
        FROM {elements_table_fqn}
        GROUP BY type
        ORDER BY n DESC
        """
    )
)

# COMMAND ----------

display(
    spark.sql(
        f"""
        SELECT filename, COUNT(*) AS n_pages
        FROM {pages_table_fqn}
        GROUP BY filename
        ORDER BY n_pages DESC
        """
    )
)

# COMMAND ----------

display(
    spark.sql(
        f"""
        SELECT filename, element_id, LEFT(description, 200) AS description_preview
        FROM {elements_table_fqn}
        WHERE type = 'figure' AND description IS NOT NULL
        LIMIT 5
        """
    )
)
