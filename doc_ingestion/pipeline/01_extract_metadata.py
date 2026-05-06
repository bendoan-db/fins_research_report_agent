# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Extract metadata and stage raw file contents via Auto Loader
# MAGIC
# MAGIC Streaming ingestion of files in the source volume into a single Delta
# MAGIC table named by `ingestion_pipeline.metadata_extraction_step.document_metadata_table`
# MAGIC in `ingestion_config.yaml`. Each row holds the filename-derived metadata
# MAGIC plus the file's raw binary contents. A downstream notebook is
# MAGIC responsible for parsing the binary into structured content via
# MAGIC `ai_parse_document`.
# MAGIC
# MAGIC AI functions applied per file:
# MAGIC - `ai_classify` v2 — pick a ticker from the configured list
# MAGIC - `ai_classify` v2 — pick a quarter from `{1, 2, 3, 4, annual}`
# MAGIC - `ai_extract`  v2 — pull the 4-digit calendar year
# MAGIC
# MAGIC Auto Loader (`cloudFiles`) tracks already-processed files via the
# MAGIC checkpoint, so re-runs only pay AI-function cost for genuinely new
# MAGIC files. No manual MERGE / dedup.
# MAGIC
# MAGIC **Compute:** Serverless SQL warehouse, Serverless compute, or DBR ≥
# MAGIC 16.4 shared. AI v2 functions are not available on standard interactive
# MAGIC clusters.
# MAGIC
# MAGIC **Re-uploads:** Auto Loader is path-based; a file replaced in place at
# MAGIC the same path will not be re-ingested. Delete the target row + the
# MAGIC checkpoint entry to force reprocessing.

# COMMAND ----------

import json
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

me = config["ingestion_pipeline"]["metadata_extraction_step"]
document_metadata_table = me["document_metadata_table"]

# Tickers come back as a list of {symbol, aliases?} dicts. The canonical
# symbol is implicitly an alias of itself; extra aliases are optional.
ticker_entries = me["tickers"]
tickers = [t["symbol"] for t in ticker_entries]
ticker_aliases = []
for entry in ticker_entries:
    symbol = entry["symbol"]
    ticker_aliases.append((symbol, symbol))
    for alias in entry.get("aliases") or []:
        ticker_aliases.append((alias, symbol))

volume_path = f"/Volumes/{catalog}/{schema}/{volume}"
target_table = f"{catalog}.{schema}.{document_metadata_table}"
checkpoint_path = f"{volume_path}/{me['checkpoint_prefix']}/{document_metadata_table}"

print(f"source:     {volume_path}")
print(f"target:     {target_table}")
print(f"checkpoint: {checkpoint_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build label/schema JSON for the AI functions

# COMMAND ----------

ticker_labels_json = json.dumps(tickers)
quarter_labels_json = json.dumps(["1", "2", "3", "4", "annual"])
year_schema_json = json.dumps(
    {
        "type": "object",
        "properties": {
            "year": {
                "type": "integer",
                "description": (
                    "The 4-digit calendar year referenced in the filename. "
                    "Encodings to handle: FY24 or fy24 means 2024; "
                    "Q124 means Q1 of 2024; Q425 means Q4 of 2025; "
                    "2025q1 means 2025. Always return a 4-digit year between "
                    "2000 and 2100."
                ),
            }
        },
        "required": ["year"],
    }
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build regex-first metadata expressions
# MAGIC
# MAGIC Every filename in this corpus follows one of a small set of mechanical
# MAGIC patterns, so deterministic regex extraction is preferred. The AI v2
# MAGIC functions are only invoked as a `COALESCE` fallback when no pattern
# MAGIC matches — Spark short-circuits `COALESCE`, so AI calls happen only on
# MAGIC genuine miss.
# MAGIC
# MAGIC Patterns covered:
# MAGIC - **Ticker** — case-insensitive substring match against canonical
# MAGIC   tickers plus `alphabet`→GOOG and `NVIDIA`→NVDA aliases.
# MAGIC - **Quarter** — `Q[1-4]YY` (AMZN), `FYYYQ[1-4]` (MSFT), `YYYY[qQ][1-4]`
# MAGIC   (alphabet), `annual.report` → `annual` (NVIDIA).
# MAGIC - **Year** — 4-digit `20XX`; or 2-digit `YY` extracted from
# MAGIC   `Q[1-4]YY` / `FYYY` and prefixed with `20`.

# COMMAND ----------

filename_expr = "regexp_extract(path, '([^/]+)$', 1)"

# Ticker: deterministic substring (from config) → ai_classify fallback.
ticker_case = " ".join(
    f"WHEN {filename_expr} ILIKE '%{alias}%' THEN '{ticker}'"
    for alias, ticker in ticker_aliases
)
ticker_fallback = (
    f"CAST(variant_get(ai_classify({filename_expr}, '{ticker_labels_json}', "
    f"map('version', '2.0')), '$.response[0]', 'STRING') AS STRING)"
)
ticker_expr = f"COALESCE(CASE {ticker_case} END, {ticker_fallback})"

# Quarter: three quarter-encodings + annual marker → ai_classify fallback.
quarter_fallback = (
    f"CAST(variant_get(ai_classify({filename_expr}, '{quarter_labels_json}', "
    f"map('version', '2.0')), '$.response[0]', 'STRING') AS STRING)"
)
quarter_expr = (
    "COALESCE("
    f"NULLIF(regexp_extract({filename_expr}, '[Qq]([1-4])[0-9]{{2}}', 1), ''),"
    f"NULLIF(regexp_extract({filename_expr}, '[Ff][Yy][0-9]{{2}}[Qq]([1-4])', 1), ''),"
    f"NULLIF(regexp_extract({filename_expr}, '[0-9]{{4}}[Qq]([1-4])', 1), ''),"
    f"CASE WHEN {filename_expr} RLIKE '(?i)annual[ _-]?report' THEN 'annual' END,"
    f"{quarter_fallback}"
    ")"
)

# Year: 4-digit, then 2-digit forms with '20' prefix → ai_extract fallback.
year_fallback = (
    f"CAST(variant_get(ai_extract({filename_expr}, '{year_schema_json}', "
    f"map('version', '2.0')), '$.response.year', 'INT') AS INT)"
)
year_expr = (
    "COALESCE("
    f"TRY_CAST(NULLIF(regexp_extract({filename_expr}, '(20[0-9]{{2}})', 1), '') AS INT),"
    f"TRY_CAST(CONCAT('20', NULLIF(regexp_extract({filename_expr}, '[Qq][1-4]([0-9]{{2}})', 1), '')) AS INT),"
    f"TRY_CAST(CONCAT('20', NULLIF(regexp_extract({filename_expr}, '[Ff][Yy]([0-9]{{2}})', 1), '')) AS INT),"
    f"{year_fallback}"
    ")"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stream from the volume, derive metadata, stage raw binary contents
# MAGIC
# MAGIC `binaryFile` source gives `path`, `modificationTime`, `length`,
# MAGIC `content` (BINARY). The filename feeds the classifier/extractor and
# MAGIC `content` is passed through unchanged.

# COMMAND ----------

stream = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("pathGlobFilter", "*.{pdf,pptx}")
    .load(volume_path)
    .selectExpr(
        "path",
        f"{filename_expr} AS filename",
        f"({ticker_expr}) AS ticker",
        f"({quarter_expr}) AS quarter",
        f"({year_expr}) AS year",
        "modificationTime AS modification_time",
        "length AS size_bytes",
        "content",
    )
)

query = (
    stream.writeStream.option("checkpointLocation", checkpoint_path)
    .trigger(availableNow=True)
    .toTable(target_table)
)

query.awaitTermination()
total_processed = sum(p.get("numInputRows", 0) for p in query.recentProgress)
print(f"Stream finished. {total_processed} new file(s) processed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

display(
    spark.sql(
        f"""
        SELECT ticker, quarter, year, COUNT(*) AS n
        FROM {target_table}
        GROUP BY ALL
        ORDER BY ticker, year, quarter
        """
    )
)

# COMMAND ----------

display(
    spark.sql(
        f"""
        SELECT
          filename, ticker, quarter, year, size_bytes,
          length(content) AS content_bytes
        FROM {target_table}
        ORDER BY ticker, year, quarter, filename
        """
    )
)
