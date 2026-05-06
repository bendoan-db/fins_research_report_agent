"""Tools exposed to the agent.

Each section subagent picks its own subset of these (declared in
`agent_config.yaml:subagents[].tools`); the assembler has `tools: []`.
The `_TOOL_REGISTRY` at the bottom of this module is the authoritative
name → callable mapping that `agent.py:_build_section_agents` reads.

Tool inventory:
  - `search_earnings_docs`        — single-query semantic search.
  - `multi_query_search`          — batch several semantic queries at once,
                                    dedupe on `(filename, quarter, year,
                                    content[:80])` so multi-angle coverage
                                    costs one LLM tool turn instead of N.
  - `list_available_periods`      — SQL: distinct (quarter, year, filename)
                                    rows for the ticker. Lets a subagent
                                    discover the time series before drilling
                                    in.
  - `get_period_full_content`     — SQL: every page of a given (ticker,
                                    quarter, year) returned in `page_id`
                                    order. Use when semantic top-k isn't
                                    enough and the subagent needs the full
                                    filing for one period.
  - `save_report_to_volume`       — write the assembled markdown to a UC
                                    volume; called deterministically from a
                                    graph node (not exposed to subagents).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementParameterListItem, StatementState
from databricks_langchain import DatabricksVectorSearch
from langchain_core.tools import tool

from agent_server.config import load_agent_config

logger = logging.getLogger(__name__)


def _resolve_vs_client_args() -> dict:
    """Resolve a usable host + bearer token for VectorSearchClient. Works for
    all auth methods the SDK can resolve — PAT, SP OAuth M2M (Databricks Apps
    runtime), and developer OAuth U2M (`databricks auth login`).

    `databricks_langchain.DatabricksVectorSearch` only auto-passes credentials
    when `auth_type in ('pat', 'oauth-m2m')`; for `databricks-cli` (local
    dev) we feed an extracted bearer token in via `client_args` instead.
    """
    cfg = WorkspaceClient().config
    headers = cfg.authenticate()
    auth_value = headers.get("Authorization", "")
    if not auth_value.startswith("Bearer "):
        raise RuntimeError(
            "Databricks SDK could not produce a bearer token for "
            "VectorSearchClient. Configure DATABRICKS_TOKEN, DATABRICKS_CLIENT_ID/"
            "DATABRICKS_CLIENT_SECRET, or `databricks auth login --profile <p>`."
        )
    return {
        "workspace_url": cfg.host,
        "personal_access_token": auth_value.removeprefix("Bearer "),
        "disable_notice": True,
    }


@lru_cache(maxsize=1)
def _retriever() -> DatabricksVectorSearch:
    """Build the vector-search retriever once per process. The `columns`
    requested from the index come from `retrieval_config.columns` in
    `agent_config.yaml` — add a column there to surface it in tool results.

    NOTE: `page_id` and `n_elements` are intentionally absent from the
    default column list because the initial index was created with default
    `columns_to_sync` and the default subset omits some INT columns. Once
    the index is rebuilt via 06a with explicit `columns_to_sync`, those
    can be added in yaml without code changes.
    """
    cfg = load_agent_config()
    rc = cfg["retrieval_config"]
    db = cfg["databricks_config"]
    index_full_name = f"{db['catalog']}.{db['schema']}.{rc['vector_search_index']}"
    return DatabricksVectorSearch(
        endpoint=rc["vector_search_endpoint"],
        index_name=index_full_name,
        # text_column is implicit for managed-embedding Delta-sync indexes;
        # the wrapper resolves it from the index's `embedding_source_columns`.
        columns=list(rc["columns"]),
        client_args=_resolve_vs_client_args(),
    )


# Bind the default top-k from yaml at module load so the tool's schema (the
# `k` parameter the LLM sees) reflects the configured default. Section
# subagents can still override per-call by passing `k=N`; this just sets
# the fallback when they don't.
_TOP_K_DEFAULT: int = int(load_agent_config()["retrieval_config"].get("top_k", 3))


@tool
def search_earnings_docs(
    query: str,
    ticker: str,
    k: int = _TOP_K_DEFAULT,
) -> list[dict[str, Any]]:
    """Semantic search over earnings slide decks and annual reports indexed
    in Databricks Vector Search. ``ticker`` is required and used as a filter
    (cross-ticker queries are not supported in v1). Returns the top-k
    matching page chunks. Each dict contains ``content`` (the page's
    rendered markdown) plus whichever metadata columns are configured in
    ``retrieval_config.columns`` of ``agent_config.yaml``.
    """
    if not ticker:
        raise ValueError("ticker is required — pass the company's stock symbol (e.g. 'AMZN').")
    docs = _retriever().similarity_search(
        query=query,
        k=k,
        filter={"ticker": ticker.upper()},
    )
    out: list[dict[str, Any]] = []
    for d in docs:
        row: dict[str, Any] = {"content": d.page_content}
        if d.metadata:
            # Pass through every metadata field the index returned — the set
            # is determined by `retrieval_config.columns` in yaml.
            row.update(d.metadata)
        out.append(row)
    return out


@tool
def multi_query_search(
    ticker: str,
    queries: list[str],
    k_per_query: int = 2,
) -> list[dict[str, Any]]:
    """Run several semantic searches over the earnings index in one tool call,
    dedupe overlapping hits, and return a merged result list. Use when you
    have a few related angles to cover (e.g. "operating margin", "free cash
    flow", "diluted EPS") and don't want to spend one tool-call turn per
    query. ``ticker`` is required and applied as a filter to every query;
    each query returns at most ``k_per_query`` chunks. Same per-row shape as
    `search_earnings_docs` (``content`` plus the configured metadata
    columns)."""
    if not ticker:
        raise ValueError("ticker is required")
    if not queries:
        return []
    seen: set[tuple] = set()
    out: list[dict[str, Any]] = []
    for q in queries:
        for row in search_earnings_docs.invoke(
            {"query": q, "ticker": ticker, "k": k_per_query}
        ):
            key = (
                row.get("filename"),
                row.get("quarter"),
                row.get("year"),
                (row.get("content") or "")[:80],
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
    return out


# ---------------------------------------------------------------------------
# SQL-warehouse-backed tools — read directly from the source `gold_documents`
# table for fully-faithful (non-semantic) period drilling.
# ---------------------------------------------------------------------------

def _warehouse_id() -> str:
    wh = os.environ.get("DATABRICKS_WAREHOUSE_ID")
    if not wh:
        raise RuntimeError(
            "DATABRICKS_WAREHOUSE_ID is not set — bind the `sql-warehouse` "
            "Apps resource in databricks.yml or export the env var locally."
        )
    return wh


def _gold_documents_fqn() -> str:
    cfg = load_agent_config()
    db = cfg["databricks_config"]
    rc = cfg["retrieval_config"]
    return f"{db['catalog']}.{db['schema']}.{rc['gold_documents_table']}"


def _exec_sql(
    statement: str,
    parameters: list[StatementParameterListItem] | None = None,
) -> list[list[Any]]:
    resp = WorkspaceClient().statement_execution.execute_statement(
        warehouse_id=_warehouse_id(),
        statement=statement,
        parameters=parameters,
        wait_timeout="30s",
    )
    if resp.status.state != StatementState.SUCCEEDED:
        err = resp.status.error
        raise RuntimeError(
            f"Statement failed ({resp.status.state}): "
            f"{err.message if err else 'unknown'}"
        )
    return (resp.result.data_array if resp.result else None) or []


@tool
def list_available_periods(ticker: str) -> list[dict[str, Any]]:
    """List the distinct (quarter, year, filename, n_pages) tuples available
    for the ticker. Run this BEFORE drilling into a specific period so you
    know what the time series looks like (e.g. "the latest quarter is Q4
    2025; we have 8 quarters going back to Q1 2024"). Useful for the
    Overview's "latest period snapshot" decision and the Financial
    Performance time-series table."""
    if not ticker:
        raise ValueError("ticker is required")
    rows = _exec_sql(
        f"""
        SELECT quarter, year, filename, COUNT(*) AS n_pages
        FROM {_gold_documents_fqn()}
        WHERE ticker = :ticker
        GROUP BY quarter, year, filename
        ORDER BY year, quarter
        """,
        parameters=[
            StatementParameterListItem(
                name="ticker", value=ticker.upper(), type="STRING"
            )
        ],
    )
    return [
        {"quarter": r[0], "year": int(r[1]), "filename": r[2], "n_pages": int(r[3])}
        for r in rows
    ]


@tool
def get_period_full_content(
    ticker: str,
    quarter: str,
    year: int,
    max_pages: int = 30,
) -> str:
    """Pull every page of a single filing (one ticker × one quarter × one
    year) as a single concatenated markdown string, ordered by page_id.
    Use when semantic top-k isn't enough and you need the full filing for
    a specific period — e.g. building the Financial Performance table from
    a single 10-Q's complete P&L. Pages are separated by `\\n\\n---\\n\\n`.
    Truncated at ``max_pages`` to bound payload size."""
    if not ticker:
        raise ValueError("ticker is required")
    rows = _exec_sql(
        f"""
        SELECT page_id, content_md
        FROM {_gold_documents_fqn()}
        WHERE ticker = :ticker AND quarter = :quarter AND year = :year
        ORDER BY page_id
        LIMIT :max_pages
        """,
        parameters=[
            StatementParameterListItem(
                name="ticker", value=ticker.upper(), type="STRING"
            ),
            StatementParameterListItem(name="quarter", value=str(quarter), type="STRING"),
            StatementParameterListItem(name="year", value=str(int(year)), type="INT"),
            StatementParameterListItem(
                name="max_pages", value=str(int(max_pages)), type="INT"
            ),
        ],
    )
    if not rows:
        return f"_(no pages indexed for {ticker} {quarter} {year})_"
    return "\n\n---\n\n".join(str(r[1]) for r in rows)


# ---------------------------------------------------------------------------
# Save-to-volume tool (called from the `save_report` graph node, not
# exposed to subagents).
# ---------------------------------------------------------------------------

@tool
def save_report_to_volume(markdown: str, ticker: str, save_location: str) -> str:
    """Save the final research-report markdown to a Unity Catalog volume.

    Final path is ``{save_location.rstrip('/')}/{ticker}_{utc_iso}.md`` —
    e.g. ``/Volumes/.../research_reports/AMZN_20260506T143045Z.md``. The
    deployed app's service principal must have ``WRITE VOLUME`` on the
    target volume; locally, whatever auth the SDK resolves is used.

    Returns the full file path written, or ``"ERROR: <reason>"`` on
    failure (we never raise — a permission issue shouldn't fail the whole
    request when the report itself was produced successfully).
    """
    if not save_location.startswith("/Volumes/"):
        return f"ERROR: save_location must start with /Volumes/ (got {save_location!r})"
    base = save_location.rstrip("/")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    file_path = f"{base}/{ticker}_{timestamp}.md"
    try:
        WorkspaceClient().files.upload(
            file_path,
            BytesIO(markdown.encode("utf-8")),
            overwrite=True,
        )
        return file_path
    except Exception as exc:  # noqa: BLE001 — surface in saved_path, don't kill the request
        logger.warning("Failed to save report to %s: %s", file_path, exc)
        return f"ERROR: {exc}"


# ---------------------------------------------------------------------------
# Registry — maps the names used in `agent_config.yaml:subagents[].tools`
# to the actual @tool callables. `save_report_to_volume` is intentionally
# NOT in here — it's called from the `save_report` graph node, never
# exposed to a subagent's LLM loop.
# ---------------------------------------------------------------------------

SECTION_TOOL_REGISTRY: dict[str, Any] = {
    "search_earnings_docs": search_earnings_docs,
    "multi_query_search": multi_query_search,
    "list_available_periods": list_available_periods,
    "get_period_full_content": get_period_full_content,
}
