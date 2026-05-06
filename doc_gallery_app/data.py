"""Cached SQL access against the gold gallery table via the Databricks SDK."""

from __future__ import annotations

import json
import os
from typing import Any

import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID") or "995bc1ffd2ff99a4"
GALLERY_TABLE = (
    os.environ.get("GALLERY_TABLE") or "doan.difficult_doc_qa.gold_documents_gallery"
)


@st.cache_resource
def workspace_client() -> WorkspaceClient:
    return WorkspaceClient()


def _query(sql: str) -> list[list[Any]]:
    resp = workspace_client().statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=sql,
        wait_timeout="50s",
    )
    if resp.status.state != StatementState.SUCCEEDED:
        err = resp.status.error
        msg = err.message if err else "unknown"
        raise RuntimeError(f"Statement failed ({resp.status.state}): {msg}")
    return (resp.result.data_array if resp.result else None) or []


@st.cache_data(ttl=300)
def list_documents() -> list[tuple[str, int]]:
    rows = _query(
        f"""
        SELECT filename, COUNT(*) AS n_pages
        FROM {GALLERY_TABLE}
        GROUP BY filename
        ORDER BY filename
        """
    )
    return [(r[0], int(r[1])) for r in rows]


@st.cache_data(ttl=300)
def list_pages(filename: str) -> list[int]:
    fn_lit = filename.replace("'", "''")
    rows = _query(
        f"""
        SELECT page_id FROM {GALLERY_TABLE}
        WHERE filename = '{fn_lit}'
        ORDER BY page_id
        """
    )
    return [int(r[0]) for r in rows]


@st.cache_data(ttl=300)
def get_page(filename: str, page_id: int) -> dict | None:
    fn_lit = filename.replace("'", "''")
    rows = _query(
        f"""
        SELECT
          filename,
          ticker,
          quarter,
          year,
          page_id,
          n_elements,
          image_uri,
          content_md,
          summary,
          to_json(elements) AS elements_json
        FROM {GALLERY_TABLE}
        WHERE filename = '{fn_lit}' AND page_id = {int(page_id)}
        LIMIT 1
        """
    )
    if not rows:
        return None
    r = rows[0]
    return {
        "filename": r[0],
        "ticker": r[1],
        "quarter": r[2],
        "year": int(r[3]) if r[3] is not None else None,
        "page_id": int(r[4]),
        "n_elements": int(r[5]) if r[5] is not None else 0,
        "image_uri": r[6],
        "content_md": r[7],
        "summary": r[8],
        "elements": json.loads(r[9]) if r[9] else [],
    }
