"""Streamlit gallery for browsing pages from the doc_ingestion pipeline."""

from __future__ import annotations

import streamlit as st

from data import GALLERY_TABLE, get_page, list_documents, list_pages
from image import color_for, load_image_bytes, render_with_bboxes

st.set_page_config(page_title="Doc Gallery", layout="wide")


def _format_period(quarter: str | None, year: int | None) -> str:
    if year is None:
        return ""
    if quarter == "annual":
        return f"Annual {year}"
    if quarter:
        return f"Q{quarter} {year}"
    return str(year)


def md_escape(text: str | None) -> str:
    """Escape dollar signs so Streamlit's markdown renderer doesn't treat them
    as LaTeX math delimiters — matters for any prose containing financial
    figures like ``$36.15 billion``. Only call on text that will be rendered
    via st.markdown / st.caption (NOT on raw HTML, which is rendered verbatim
    via unsafe_allow_html)."""
    if not text:
        return text or ""
    return text.replace("$", r"\$")


# ----------------------------------------------------------------------------
# Sidebar — document and page picker
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Page picker")

    docs = list_documents()
    if not docs:
        st.error(f"No rows in {GALLERY_TABLE}.")
        st.stop()

    doc_idx = st.selectbox(
        "Document",
        options=list(range(len(docs))),
        format_func=lambda i: f"{docs[i][0]}  ({docs[i][1]} pages)",
    )
    selected_filename = docs[doc_idx][0]

    pages = list_pages(selected_filename)
    if not pages:
        st.error(f"No pages for {selected_filename}.")
        st.stop()

    if len(pages) > 12:
        page_id = st.slider(
            "Page",
            min_value=min(pages),
            max_value=max(pages),
            value=pages[0],
            step=1,
        )
        # Snap to a value that actually exists (ranges may have gaps).
        if page_id not in pages:
            page_id = min(pages, key=lambda p: abs(p - page_id))
    else:
        page_id = st.selectbox("Page", options=pages)

    st.caption(f"Source table: `{GALLERY_TABLE}`")


# ----------------------------------------------------------------------------
# Load the selected page
# ----------------------------------------------------------------------------
page = get_page(selected_filename, page_id)
if page is None:
    st.error(f"Page not found: {selected_filename} p{page_id}.")
    st.stop()

period = _format_period(page["quarter"], page["year"])

st.title(f"{page['filename']} — page {page['page_id']}")
caption_bits = [page["ticker"], period, f"{page['n_elements']} elements"]
st.caption(" · ".join(b for b in caption_bits if b))


# ----------------------------------------------------------------------------
# Main area — image + right-side inspector
# ----------------------------------------------------------------------------
left, right = st.columns([3, 2], gap="large")

with left:
    if not page.get("image_uri"):
        st.warning("This page has no image_uri.")
    else:
        try:
            img_bytes = load_image_bytes(page["image_uri"])
            rendered = render_with_bboxes(img_bytes, page["elements"])
            st.image(rendered, use_column_width=True)
        except Exception as exc:  # noqa: BLE001 — render errors are user-visible
            st.error(f"Failed to render image: {exc}")

with right:
    tab_elements, tab_md, tab_summary = st.tabs(["Elements", "Markdown", "Summary"])

    with tab_elements:
        elements = page["elements"]
        if not elements:
            st.info("No elements on this page.")
        for el in elements:
            elem_type = el.get("type", "") or ""
            color = color_for(elem_type)
            confidence = el.get("confidence")
            confidence_html = (
                f"<span style='float:right;font-variant-numeric:tabular-nums;"
                f"color:#6b7280;font-size:0.85em;'>conf {confidence:.0%}</span>"
                if isinstance(confidence, (int, float))
                else ""
            )
            with st.container(border=True):
                st.markdown(
                    (
                        f"<span style='display:inline-block;width:12px;height:12px;"
                        f"background:{color};border-radius:2px;margin-right:8px;"
                        f"vertical-align:middle;'></span>"
                        f"<strong>#{el.get('element_id', '?')} {elem_type}</strong>"
                        f"{confidence_html}"
                    ),
                    unsafe_allow_html=True,
                )
                description = el.get("description")
                if description:
                    st.caption(md_escape(description))
                caption = el.get("caption")
                if caption:
                    with st.expander("Caption", expanded=False):
                        st.markdown(md_escape(caption))
                content = el.get("content") or ""
                if content:
                    if elem_type == "table":
                        st.markdown(
                            f"<div style='overflow-x:auto;font-size:0.85em;'>"
                            f"<style>"
                            f".element-table table{{border-collapse:collapse;width:100%;}}"
                            f".element-table th,.element-table td{{"
                            f"border:1px solid rgba(125,125,125,0.4);"
                            f"padding:4px 8px;text-align:left;}}"
                            f".element-table th{{"
                            f"background:rgba(125,125,125,0.15);"
                            f"font-weight:600;}}"
                            f"</style>"
                            f"<div class='element-table'>{content}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    elif len(content) > 300:
                        with st.expander(f"{content[:300]}…", expanded=False):
                            st.markdown(md_escape(content))
                    else:
                        st.markdown(md_escape(content))

    with tab_md:
        st.markdown(md_escape(page["content_md"]) or "_(empty)_")

    with tab_summary:
        st.markdown(md_escape(page["summary"]) or "_(no summary)_")
