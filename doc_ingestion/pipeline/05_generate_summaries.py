# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Generate page summaries and emit `gold_documents`
# MAGIC
# MAGIC For each row in `silver_document_elements_aggregated`, call `ai_query`
# MAGIC against the configured chat endpoint with a concise retrieval-style
# MAGIC prompt, conditioned on the page's `ticker` / `quarter` / `year` and its
# MAGIC `content_md`. Compose `content_to_embed` = `content_md` + an appended
# MAGIC `## Summary` block, and join the corresponding page `image_uri` from
# MAGIC `silver_document_pages`. Output table:
# MAGIC `<catalog>.<schema>.<gold_documents_table>` (default `gold_documents`).
# MAGIC
# MAGIC Pipeline shape:
# MAGIC
# MAGIC 1. Stream `silver_document_elements_aggregated` (`Trigger.AvailableNow`).
# MAGIC 2. Static-join with `silver_document_pages` on `(path, page_id)` to
# MAGIC    fetch `image_uri`.
# MAGIC 3. Inline `ai_query` in `selectExpr` with `failOnError => false` so a
# MAGIC    single bad row doesn't kill the stream — failures yield NULL.
# MAGIC 4. Second `selectExpr` derives `content_to_embed` via
# MAGIC    `coalesce(concat(content_md, '\n\n## Summary\n\n', summary), content_md)`
# MAGIC    so a NULL summary falls back to the raw content.
# MAGIC 5. `writeStream.toTable` (append) with a per-table checkpoint.
# MAGIC
# MAGIC **Run order**: must follow 04 so the aggregated source is populated.
# MAGIC
# MAGIC **Compute**: serverless. The endpoint configured in
# MAGIC `ingestion_config.yaml` (`databricks-claude-sonnet-4` by default) must
# MAGIC be deployed and accessible. No image input — pure text in, text out.

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
pa = config["ingestion_pipeline"]["page_aggregation_step"]
ss = config["ingestion_pipeline"]["summarization_step"]

pages_table = dp["parsed_document_pages_table"]
agg_table = pa["aggregated_document_pages_table"]
gold_table = ss["gold_documents_table"]

source_volume_path = f"/Volumes/{catalog}/{schema}/{volume}"
pages_table_fqn = f"{catalog}.{schema}.{pages_table}"
agg_table_fqn = f"{catalog}.{schema}.{agg_table}"
gold_table_fqn = f"{catalog}.{schema}.{gold_table}"
gold_checkpoint = f"{source_volume_path}/{ss['gold_documents_checkpoint_prefix']}/{gold_table}"

summary_endpoint = ss["summary_endpoint"]
summary_prompt = " ".join(ss["summarization_prompt"].split())

print(f"aggregated:  {agg_table_fqn}")
print(f"pages:       {pages_table_fqn}")
print(f"gold:        {gold_table_fqn}")
print(f"checkpoint:  {gold_checkpoint}")
print(f"endpoint:    {summary_endpoint}")
print(f"prompt:      {summary_prompt[:120]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build the gold stream
# MAGIC
# MAGIC Static-read the pages table (small) so each streaming microbatch can
# MAGIC look up image URIs without paying a per-row cost. Then call `ai_query`
# MAGIC inline in a `selectExpr`. The summary call is split from
# MAGIC `content_to_embed` into two `selectExpr` stages because Spark SQL can't
# MAGIC reference a sibling alias from the same `SELECT`.

# COMMAND ----------

prompt_sql = summary_prompt.replace("'", "''")

# Period rendering — quarterly tickers are 1..4, NVDA's annual filings use 'annual'.
period_expr = (
    "CASE "
    "  WHEN quarter = 'annual' THEN concat('Annual ', CAST(year AS STRING)) "
    "  ELSE concat('Q', quarter, ' ', CAST(year AS STRING)) "
    "END"
)

pages_static = (
    spark.read.table(pages_table_fqn)
    .selectExpr("path AS p_path", "page_id AS p_page_id", "image_uri")
)

stream_with_summary = (
    spark.readStream.table(agg_table_fqn)
    .join(
        pages_static,
        (col("path") == col("p_path")) & (col("page_id") == col("p_page_id")),
        "left",
    )
    .drop("p_path", "p_page_id")
    .selectExpr(
        "path",
        "filename",
        "ticker",
        "quarter",
        "year",
        "page_id",
        "uuid",
        "n_elements",
        "content_md",
        "image_uri",
        f"""
        ai_query(
          '{summary_endpoint}',
          concat(
            '{prompt_sql}',
            '\\n\\nTicker: ', ticker,
            ' | Period: ', {period_expr},
            '\\n\\nContent:\\n\\n', content_md
          ),
          failOnError => false
        ).result AS summary
        """,
    )
)

gold_stream = stream_with_summary.selectExpr(
    "path",
    "filename",
    "ticker",
    "quarter",
    "year",
    "page_id",
    "uuid",
    "n_elements",
    "content_md",
    "image_uri",
    "coalesce(concat(content_md, '\\n\\n## Summary\\n\\n', summary), content_md) AS content_to_embed",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write gold_documents

# COMMAND ----------

query = (
    gold_stream.writeStream.option("checkpointLocation", gold_checkpoint)
    .trigger(availableNow=True)
    .toTable(gold_table_fqn)
)
query.awaitTermination()
gold_input = sum(p.get("numInputRows", 0) for p in query.recentProgress)
print(f"Wrote {gold_input} new gold row(s).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

display(
    spark.sql(
        f"""
        SELECT
          ticker,
          COUNT(*)                                              AS n_pages,
          SUM(CASE WHEN image_uri IS NULL THEN 1 ELSE 0 END)    AS n_null_image,
          SUM(CASE WHEN content_to_embed = content_md THEN 1 ELSE 0 END) AS n_summary_missing,
          ROUND(AVG(LENGTH(content_to_embed)))                  AS avg_to_embed_chars
        FROM {gold_table_fqn}
        GROUP BY ticker
        ORDER BY ticker
        """
    )
)

# COMMAND ----------

display(
    spark.sql(
        f"""
        SELECT filename, page_id, RIGHT(content_to_embed, 700) AS tail_preview
        FROM {gold_table_fqn}
        WHERE filename LIKE '2025q1-alphabet%'
        ORDER BY page_id
        LIMIT 3
        """
    )
)
