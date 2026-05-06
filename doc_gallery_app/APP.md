# `doc_gallery_app` — design + plan

## Purpose

A small web UI for inspecting the output of the `doc_ingestion` pipeline. For any page in any document the user can see, side by side:

- the **source page image** (the slide as rendered by `ai_parse_document`),
- **bounding box overlays** for every extracted element on that page,
- the **per-element details** (type, content, description, caption) in a list, and
- the **per-page rendered markdown** (`content_md`) so the parsed output can be eyeballed against the source.

Single source of truth: the `doc.difficult_doc_qa.gold_documents_gallery` table built by `doc_ingestion/pipeline/06_build_gallery.py`. All other tables in the pipeline are reachable transitively from this one (path, page_id, image_uri, content_md, summary, and the `elements ARRAY<STRUCT<…>>` payload).

## Requirements

1. **Document/page picker** in a left sidebar. Two-level: pick document by filename → pick page by `page_id` (with the page count for the document visible). Default to the first document, page 0 on cold start.
2. **Central viewer** that renders the page image at a sensible size (fit-to-width, maintain aspect ratio) with **colored bounding box rectangles overlaid** for each element on that page. Each rectangle should be labeled with at least the `element_id` and `type`.
3. **Right panel** that shows, for the selected page:
   - One row per element: `element_id`, `type`, a short preview of `content` (or `caption` for figures), and `description` if present. Color-coded to match the bbox color in the central viewer.
   - The full `content_md` markdown for the page rendered as markdown (separate section, beneath or above the element list).
4. **No edits, no writes**: read-only browser of the gold gallery table. No row updates, no annotation persistence (out of scope for v1).
5. **Runs as a Databricks App** using the same workspace as the pipeline (`fe-vm-vdm-classic-hkbucz`) so it can read the volume-mounted page images directly.

## Source data — `doan.difficult_doc_qa.gold_documents_gallery`

```
path        STRING     -- /Volumes/.../earnings_slides/<file>
filename    STRING
ticker      STRING
quarter     STRING     -- '1'..'4' or 'annual'
year        INT
page_id     INT        -- 0-indexed page number within the document
n_elements  INT
image_uri   STRING     -- /Volumes/.../ai_parse_doc_images/<...>.jpg
content_md  STRING     -- full page markdown rendered by step 04
summary     STRING     -- AI-generated summary appended in step 05
elements    ARRAY<STRUCT<
              element_id  INT,
              type        STRING,    -- 'title' | 'section_header' | 'text' | 'figure' | 'table' | 'page_header' | …
              content     STRING,
              description STRING,    -- ai_parse_document description (set for figures)
              bbox        VARIANT,   -- [{coord:[x1,y1,x2,y2], page_id:INT}, …] (length-1 in practice)
              caption     STRING     -- non-null for figure rows; from silver_figure_captions
            >>
```

Row count: 642. The bbox coordinates are in **page-image pixel space** (origin top-left, x→right, y→down) — no transform needed before drawing.

## Tech stack (recommended)

| Concern              | Choice                                                                 |
| -------------------- | ---------------------------------------------------------------------- |
| Framework            | **Streamlit** (single-file Python app, fits Databricks Apps cleanly)    |
| Layout primitives    | `st.sidebar` + `st.columns([3, 2])` for the main 2-pane split          |
| Image overlay        | **Server-side PIL rendering**: download the JPEG via the SDK, draw rectangles + labels with `PIL.ImageDraw`, hand the modified bytes to `st.image` |
| Markdown rendering   | `st.markdown` (already supports headers, tables, etc.)                  |
| Data access          | `databricks-sdk` `WorkspaceClient().statement_execution.execute_statement(...)` for SQL and `WorkspaceClient().files.download(...)` for page images — one dependency, one auth path |
| Auth                 | Databricks Apps runtime auto-injects `DATABRICKS_HOST` + `DATABRICKS_CLIENT_ID` + `DATABRICKS_CLIENT_SECRET` for the app's service principal; `WorkspaceClient()` picks these up automatically |
| Deployment           | `app.yaml` declares the run command + env, resources are bound via the root `databricks.yml` `resources.apps.doc_gallery_app` block; ship via `databricks bundle deploy --profile DEFAULT` |

**Why Streamlit over Dash**: the user explicitly didn't ask for hover/click interactivity between bboxes and the right panel. Pre-rendering the overlay image server-side keeps the code 100-200 lines instead of needing a Dash plotly figure with shapes and callbacks. If interactive correlation later becomes desirable, the upgrade path is to swap the pre-rendered image for a `streamlit-image-coordinates` or Dash component without changing the data layer.

**Why the SDK Files API over FUSE**: Databricks Apps run as a managed serving container. The volume FUSE mount may not be available; the SDK Files API works regardless and is the documented pattern.

## Bounding-box color map (initial)

Match the markdown header style introduced in step 04 so the visual mapping is consistent with the rendered text:

| element type     | color           | rectangle width |
| ---------------- | --------------- | --------------- |
| `title`          | red `#dc2626`   | 4 px            |
| `section_header` | orange `#ea580c`| 3 px            |
| `figure`         | blue `#2563eb`  | 3 px            |
| `table`          | green `#16a34a` | 3 px            |
| `text`           | gray `#6b7280`  | 2 px            |
| anything else    | dim gray `#9ca3af` | 2 px         |

Label placement: top-left corner of the box, format `#{element_id} {type}`, white text on a 50%-opaque colored fill. Drawn after all rectangles so labels always sit on top.

## Implementation steps

### Phase 0 — scaffolding

1. Create `doc_gallery_app/requirements.txt` with `streamlit`, `databricks-sql-connector`, `databricks-sdk`, `Pillow`, `pyyaml`.
2. Create `doc_gallery_app/app.yaml` (Databricks Apps manifest) declaring the entry point (`streamlit run app.py`) and any runtime env. Reference the existing serverless SQL warehouse and the page-image volume.
3. Create `doc_gallery_app/app.py` (single-file entry point) with the layout skeleton: sidebar + 2-column main area, no data wired up yet.

### Phase 1 — data layer

4. Add a small `data.py` module with three cached helpers (`@st.cache_data` w/ TTL):
   - `list_documents() -> list[str]` — `SELECT DISTINCT filename, COUNT(DISTINCT page_id) AS n_pages FROM gold_documents_gallery GROUP BY filename ORDER BY filename`.
   - `list_pages(filename: str) -> list[int]` — page_ids available for a doc.
   - `get_page(filename: str, page_id: int) -> dict` — full row including `image_uri`, `content_md`, `summary`, and the deserialized `elements` array (cast `bbox` VARIANT to JSON via `to_json` in the SQL).
5. Add `image.py` with:
   - `load_image_bytes(image_uri: str) -> bytes` — uses `WorkspaceClient().files.download(...)`, also `@st.cache_data` keyed on `image_uri`.
   - `render_with_bboxes(image_bytes: bytes, elements: list[dict]) -> bytes` — opens with PIL, iterates elements, draws colored rectangles + labels per the color map above, returns PNG bytes.

### Phase 2 — UI assembly

6. Sidebar:
   - Document selector (`st.selectbox`) sourced from `list_documents()`. Display `filename — N pages`.
   - Page slider or selectbox (`st.slider` if N > 12, else `st.selectbox`) sourced from `list_pages(selected_filename)`.
   - Selected `(filename, page_id)` flows to the main area via `st.session_state`.
7. Main area, left column (`st.columns([3, 2])` left = 3):
   - Title with filename + ticker + period (formatted `Q{quarter} {year}` or `Annual {year}` for NVDA-style annuals).
   - Image rendered via `st.image(render_with_bboxes(...))` at `use_container_width=True`.
8. Main area, right column:
   - Tabs: **Elements** | **Markdown** | **Summary**.
   - Elements tab: scrollable list of element cards. Each card shows the colored swatch (matching the bbox color), `#{element_id} {type}`, `description` (if any), `caption` (for figures), and `content` truncated to ~300 chars with an `expander` to read the full text.
   - Markdown tab: `st.markdown(content_md, unsafe_allow_html=False)`.
   - Summary tab: `st.markdown(summary)`.

### Phase 3 — Databricks Apps deploy wiring

The root `databricks.yml` declares a `resources.apps.doc_gallery_app` block that points at this directory (`source_code_path: ./doc_gallery_app`) and binds the SQL warehouse via `${var.sql_warehouse_id}` (defaulted to `995bc1ffd2ff99a4`). The `app.yaml` here references that binding via `valueFrom: sql-warehouse`. Streamlit port/host wiring is handled automatically by the runtime — `DATABRICKS_APP_PORT` is auto-injected and Streamlit is recognized as a supported framework.

9. **Validate the bundle**: `databricks bundle validate --profile DEFAULT` — should report the `doc-gallery-app` resource alongside the existing pipeline notebooks. To override the warehouse: `databricks bundle deploy --profile DEFAULT --var sql_warehouse_id=<other_id>`.
10. **Deploy**: `databricks bundle deploy --profile DEFAULT`. First deployment creates the app + service principal; subsequent deploys sync source code only.
11. **Grant the app's service principal data access** (one-time, after first deploy reveals the SP id — visible in the workspace Apps UI or via `databricks apps get doc-gallery-app`):
    - `GRANT SELECT ON TABLE doan.difficult_doc_qa.gold_documents_gallery TO `<sp_app_id>`;`
    - `GRANT READ VOLUME ON VOLUME doan.difficult_doc_qa.ai_parse_doc_images TO `<sp_app_id>`;`
    - The `CAN_USE` warehouse permission is conferred automatically by the resource binding declared in `databricks.yml`.
12. **Smoke-test locally** before pointing users at the deployed app: `cd doc_gallery_app && pip install -r requirements.txt && streamlit run app.py`. Local auth piggybacks on whatever `databricks auth login --profile DEFAULT` configured. Page through 2–3 documents, confirm bboxes align with the source image and labels are legible.

### Phase 4 — polish (optional, post-v1)

- Color the right-panel cards with the same hex color used for the bbox.
- "Jump to first figure" button per page.
- Filter element list by type (multiselect chips above the elements list).
- Dark-mode friendly bbox colors.

## Files (current)

```
doc_gallery_app/
├── APP.md             # this document
├── app.py             # Streamlit entry point (UI assembly)
├── data.py            # cached SQL + row hydration helpers via WorkspaceClient
├── image.py           # image fetch + PIL bbox rendering
├── app.yaml           # Databricks Apps manifest (command + env + valueFrom bindings)
└── requirements.txt   # streamlit, databricks-sdk, Pillow
```

The root `databricks.yml` declares the corresponding `resources.apps.doc_gallery_app` block and the `sql_warehouse_id` variable (defaulted to `995bc1ffd2ff99a4`).

## Open questions / decisions to confirm before coding

1. **Framework lock-in** — Streamlit is the recommendation. If you'd rather use Dash (better hover/click between bboxes and the right panel) say so before Phase 0.
2. **Filename uniqueness** — `(filename)` alone is the document key in the spec. Across the current 19-doc corpus filenames are unique, so this is fine — but if multiple uploads ever share a filename we'd need to switch the document key to `path`.
3. **Image cache scope** — `@st.cache_data` is per-app-instance memory. With 642 page images at ~few hundred KB each, full-corpus warm cache is < 250 MB and acceptable. If we want a smaller footprint, fall back to a per-session LRU.
4. **Auth model** — assumes a Databricks App with the platform-injected service principal. If the user wants per-user "on-behalf-of" auth (different users see different data), we'd need to swap to OBO flow — out of scope for v1 unless explicitly requested.
5. **Multi-region figures** — bbox in the gallery is the full VARIANT array, not a single rectangle. v1 will draw `bbox[0].coord` only, matching the convention used in 03/04. If a future ai_parse run produces multi-region elements we'd iterate the full array.
