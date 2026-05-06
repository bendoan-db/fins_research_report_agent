# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Aggregate elements + captions into one row per page
# MAGIC
# MAGIC Joins `silver_document_elements` with `silver_figure_captions` on
# MAGIC `(path, element_id)` (left join — captions only exist for figures),
# MAGIC then aggregates per `(path, page_id)` into a single markdown-rendered
# MAGIC `content_md` STRING. Each element becomes a markdown block prefixed by
# MAGIC a header chosen by `type`:
# MAGIC
# MAGIC | type             | rendering                              |
# MAGIC | ---------------- | -------------------------------------- |
# MAGIC | `title`          | `# {content}`                          |
# MAGIC | `section_header` | `## {content}`                         |
# MAGIC | `figure`         | `### Figure\n\n{caption}`              |
# MAGIC | `table`          | `### Table\n\n{content}`               |
# MAGIC | anything else    | `{content}` (no header)                |
# MAGIC
# MAGIC Within a page, blocks are joined in `element_id` order separated by
# MAGIC `\n\n`. Output table: `<catalog>.<schema>.<aggregated_document_pages_table>`
# MAGIC (default `silver_document_elements_aggregated`).
# MAGIC
# MAGIC **Pattern**: streaming from elements with `Trigger.AvailableNow` +
# MAGIC `foreachBatch` + `MERGE INTO`. Each microbatch identifies the
# MAGIC `(path, page_id)` keys touched, re-reads the full elements + captions
# MAGIC for those pages from the static silver tables, recomputes the page's
# MAGIC markdown, and merges. This keeps the stage incremental (the checkpoint
# MAGIC tracks element-stream progress) while staying correct even if a page's
# MAGIC elements straddle multiple microbatches.
# MAGIC
# MAGIC **Run order**: must run **after** 03_caption_figures so caption rows
# MAGIC for any new figures are already present when the static read happens.
# MAGIC If captions land after a 04 run, bump
# MAGIC `aggregated_document_pages_checkpoint_prefix` (e.g. `_checkpoints/v1`
# MAGIC → `_checkpoints/v2`) to force re-aggregation.
# MAGIC
# MAGIC **Compute**: serverless. Pure SQL inside `foreachBatch`, no UDFs.

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

dp = config["ingestion_pipeline"]["document_parsing_step"]
fc = config["ingestion_pipeline"]["figure_captioning_step"]
pa = config["ingestion_pipeline"]["page_aggregation_step"]

elements_table = dp["parsed_document_elements_table"]
captions_table = fc["figure_captions_table"]
agg_table = pa["aggregated_document_pages_table"]

source_volume_path = f"/Volumes/{catalog}/{schema}/{volume}"
elements_table_fqn = f"{catalog}.{schema}.{elements_table}"
captions_table_fqn = f"{catalog}.{schema}.{captions_table}"
agg_table_fqn = f"{catalog}.{schema}.{agg_table}"
agg_checkpoint = f"{source_volume_path}/{pa['aggregated_document_pages_checkpoint_prefix']}/{agg_table}"

print(f"elements:    {elements_table_fqn}")
print(f"captions:    {captions_table_fqn}")
print(f"aggregated: {agg_table_fqn}")
print(f"checkpoint: {agg_checkpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-create the target table
# MAGIC
# MAGIC `MERGE INTO` requires the target to exist. Schema is fixed; the only
# MAGIC variable contents are within `content_md`.

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {agg_table_fqn} (
      path        STRING,
      filename    STRING,
      ticker      STRING,
      quarter     STRING,
      year        INT,
      page_id     INT,
      uuid        STRING,
      n_elements  INT,
      content_md  STRING
    ) USING DELTA
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## foreachBatch: scope recompute to affected pages, then MERGE
# MAGIC
# MAGIC The streaming source is just element `(path, page_id)` keys. Inside the
# MAGIC handler we publish those keys as a temp view, then a single SQL MERGE
# MAGIC re-reads the full elements + captions for those pages, builds blocks,
# MAGIC aggregates per page, and merges into the target.

# COMMAND ----------

def merge_pages(batch_df, batch_id):
    keys = batch_df.select("path", "page_id").dropDuplicates()
    keys.createOrReplaceTempView("affected_pages")
    spark.sql(f"""
      MERGE INTO {agg_table_fqn} t
      USING (
        WITH els AS (
          SELECT
            e.path, e.filename, e.ticker, e.quarter, e.year,
            e.element_id, e.type, e.content, e.page_id,
            c.caption
          FROM {elements_table_fqn} e
          INNER JOIN affected_pages a
            ON e.path = a.path AND e.page_id = a.page_id
          LEFT JOIN {captions_table_fqn} c
            ON e.path = c.path AND e.element_id = c.element_id
        ),
        blocks AS (
          SELECT
            path, filename, ticker, quarter, year, page_id, element_id,
            CASE
              WHEN type = 'title'          THEN concat('# ',  coalesce(content, ''))
              WHEN type = 'section_header' THEN concat('## ', coalesce(content, ''))
              WHEN type = 'figure'         THEN concat('### Figure\\n\\n', coalesce(caption, '_(no caption)_'))
              WHEN type = 'table'          THEN concat('### Table\\n\\n', coalesce(content, ''))
              ELSE coalesce(content, '')
            END AS block
          FROM els
        )
        SELECT
          path,
          ANY_VALUE(filename) AS filename,
          ANY_VALUE(ticker)   AS ticker,
          ANY_VALUE(quarter)  AS quarter,
          ANY_VALUE(year)     AS year,
          page_id,
          uuid()              AS uuid,
          COUNT(*) AS n_elements,
          array_join(
            transform(
              sort_array(collect_list(struct(element_id, block))),
              x -> x.block
            ),
            '\\n\\n'
          ) AS content_md
        FROM blocks
        GROUP BY path, page_id
      ) s
      ON t.path = s.path AND t.page_id = s.page_id
      -- Preserve uuid on update so it stays stable across reruns; only INSERT
      -- generates a fresh one.
      WHEN MATCHED THEN UPDATE SET
        filename   = s.filename,
        ticker     = s.ticker,
        quarter    = s.quarter,
        year       = s.year,
        n_elements = s.n_elements,
        content_md = s.content_md
      WHEN NOT MATCHED THEN INSERT *
    """)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stream the elements table and run the aggregator

# COMMAND ----------

agg_query = (
    spark.readStream.table(elements_table_fqn)
    .selectExpr("path", "page_id")
    .writeStream
    .option("checkpointLocation", agg_checkpoint)
    .trigger(availableNow=True)
    .foreachBatch(merge_pages)
    .start()
)
agg_query.awaitTermination()
agg_input = sum(p.get("numInputRows", 0) for p in agg_query.recentProgress)
print(f"Aggregated pages from {agg_input} new element row(s).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

display(
    spark.sql(
        f"""
        SELECT
          ticker,
          COUNT(*)                  AS n_pages,
          AVG(n_elements)           AS avg_elements_per_page,
          AVG(LENGTH(content_md))   AS avg_content_md_chars
        FROM {agg_table_fqn}
        GROUP BY ticker
        ORDER BY ticker
        """
    )
)

# COMMAND ----------

display(
    spark.sql(
        f"""
        SELECT filename, page_id, n_elements, LEFT(content_md, 800) AS content_preview
        FROM {agg_table_fqn}
        WHERE filename LIKE '2025q1-alphabet%'
        ORDER BY page_id
        LIMIT 3
        """
    )
)
