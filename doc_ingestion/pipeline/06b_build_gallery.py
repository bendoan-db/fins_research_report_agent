# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Build the per-page gallery table
# MAGIC
# MAGIC Denormalizes everything a UI gallery needs into one row per page:
# MAGIC the page `image_uri`, the rendered `content_md` and extracted `summary`,
# MAGIC and an `elements ARRAY<STRUCT<…>>` payload with each element's
# MAGIC `bbox`, `type`, `content`, ai_parse `description`, and (for figures)
# MAGIC the `caption` from `silver_figure_captions`. The gallery overlays the
# MAGIC element rectangles on the page image and shows the caption text inline.
# MAGIC
# MAGIC Output table: `<catalog>.<schema>.<gallery_table>`
# MAGIC (default `gold_documents_gallery`).
# MAGIC
# MAGIC Pipeline shape:
# MAGIC
# MAGIC 1. Pre-aggregate elements ⨝ captions per page as a static DataFrame
# MAGIC    (Spark re-evaluates the static side per microbatch).
# MAGIC 2. Stream `gold_documents` (`Trigger.AvailableNow`) — page-grain source
# MAGIC    that already carries metadata, `image_uri`, `content_md`, and
# MAGIC    `content_to_embed` (markdown body + appended `## Summary`).
# MAGIC 3. Stream-static join on `(path, page_id)` to attach the elements array.
# MAGIC 4. Project to the final gallery shape, deriving `summary` by stripping
# MAGIC    the `content_md\n\n## Summary\n\n` prefix from `content_to_embed`.
# MAGIC 5. `writeStream.toTable` (append) with a per-table checkpoint.
# MAGIC
# MAGIC **Run order**: must follow 05.
# MAGIC
# MAGIC **Compute**: serverless. No AI calls — pure SQL + a few DataFrame ops.

# COMMAND ----------

from pathlib import Path

import yaml
from pyspark.sql.functions import col

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
ss = config["ingestion_pipeline"]["summarization_step"]
gs = config["ingestion_pipeline"]["gallery_step"]

elements_table = dp["parsed_document_elements_table"]
captions_table = fc["figure_captions_table"]
gold_table = ss["gold_documents_table"]
gallery_table = gs["gallery_table"]

source_volume_path = f"/Volumes/{catalog}/{schema}/{volume}"
elements_fqn = f"{catalog}.{schema}.{elements_table}"
captions_fqn = f"{catalog}.{schema}.{captions_table}"
gold_fqn = f"{catalog}.{schema}.{gold_table}"
gallery_fqn = f"{catalog}.{schema}.{gallery_table}"
gallery_checkpoint = f"{source_volume_path}/{gs['gallery_checkpoint_prefix']}/{gallery_table}"

print(f"elements:    {elements_fqn}")
print(f"captions:    {captions_fqn}")
print(f"gold:        {gold_fqn}")
print(f"gallery:     {gallery_fqn}")
print(f"checkpoint:  {gallery_checkpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-create the target table
# MAGIC
# MAGIC Explicit schema lets us pin VARIANT inside the nested struct so empty
# MAGIC microbatches don't create a schema-less Delta table.

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {gallery_fqn} (
      path        STRING,
      filename    STRING,
      ticker      STRING,
      quarter     STRING,
      year        INT,
      page_id     INT,
      n_elements  INT,
      image_uri   STRING,
      content_md  STRING,
      summary     STRING,
      elements    ARRAY<STRUCT<
                    element_id   INT,
                    type         STRING,
                    content      STRING,
                    description  STRING,
                    confidence   DOUBLE,
                    bbox         VARIANT,
                    caption      STRING
                  >>
    ) USING DELTA
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Static elements-per-page aggregation
# MAGIC
# MAGIC Left-join captions onto elements (captions only exist for figures), then
# MAGIC group by (path, page_id) and aggregate into an array sorted by
# MAGIC `element_id`. We use `array_sort` with an explicit lambda comparator
# MAGIC because the struct contains a VARIANT (`bbox`), and `sort_array` only
# MAGIC works on fully-orderable element types.

# COMMAND ----------

elements_per_page = spark.sql(f"""
    SELECT
      e.path                    AS e_path,
      e.page_id                 AS e_page_id,
      array_sort(
        collect_list(named_struct(
          'element_id',  e.element_id,
          'type',        e.type,
          'content',     e.content,
          'description', e.description,
          'confidence',  e.confidence,
          'bbox',        e.bbox,
          'caption',     c.caption
        )),
        (a, b) -> CASE
                    WHEN a.element_id < b.element_id THEN -1
                    WHEN a.element_id > b.element_id THEN  1
                    ELSE 0
                  END
      ) AS elements
    FROM {elements_fqn} e
    LEFT JOIN {captions_fqn} c
      ON e.path = c.path AND e.element_id = c.element_id
    GROUP BY e.path, e.page_id
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stream gold_documents and join the elements array

# COMMAND ----------

# Header inserted by step 05 between content_md and the AI summary. Stripping
# this prefix from `content_to_embed` recovers the bare summary text.
SUMMARY_HEADER_LITERAL = "\\n\\n## Summary\\n\\n"

gallery_stream = (
    spark.readStream.table(gold_fqn)
    .join(
        elements_per_page,
        (col("path") == col("e_path")) & (col("page_id") == col("e_page_id")),
        "left",
    )
    .drop("e_path", "e_page_id")
    .selectExpr(
        "path",
        "filename",
        "ticker",
        "quarter",
        "year",
        "page_id",
        "n_elements",
        "image_uri",
        "content_md",
        f"""
        CASE
          WHEN content_to_embed IS NULL OR content_to_embed = content_md THEN NULL
          ELSE SUBSTRING(content_to_embed, LENGTH(content_md) + LENGTH('{SUMMARY_HEADER_LITERAL}') + 1)
        END AS summary
        """,
        "elements",
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write gold_documents_gallery

# COMMAND ----------

query = (
    gallery_stream.writeStream.option("checkpointLocation", gallery_checkpoint)
    .trigger(availableNow=True)
    .toTable(gallery_fqn)
)
query.awaitTermination()
gallery_input = sum(p.get("numInputRows", 0) for p in query.recentProgress)
print(f"Wrote {gallery_input} new gallery row(s).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

display(
    spark.sql(
        f"""
        SELECT
          ticker,
          COUNT(*)                                                       AS n_pages,
          SUM(CASE WHEN image_uri IS NULL THEN 1 ELSE 0 END)             AS n_null_image,
          SUM(CASE WHEN summary   IS NULL THEN 1 ELSE 0 END)             AS n_null_summary,
          SUM(CASE WHEN elements  IS NULL OR size(elements) = 0
                     THEN 1 ELSE 0 END)                                  AS n_empty_elements,
          ROUND(AVG(size(elements)), 1)                                  AS avg_elements_per_page
        FROM {gallery_fqn}
        GROUP BY ticker
        ORDER BY ticker
        """
    )
)

# COMMAND ----------

display(
    spark.sql(
        f"""
        SELECT
          filename,
          page_id,
          e.element_id,
          e.type,
          to_json(e.bbox)              AS bbox_json,
          LEFT(e.caption, 120)         AS caption_preview
        FROM {gallery_fqn}
        LATERAL VIEW explode(elements) AS e
        WHERE filename LIKE '2025q1-alphabet%'
          AND page_id = 3
        ORDER BY e.element_id
        """
    )
)
