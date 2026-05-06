"""Page-image fetch and bbox overlay rendering."""

from __future__ import annotations

import io

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from data import workspace_client

# Color map mirrors the markdown header style used in step 04 so the visual
# mapping between bboxes and the rendered text is consistent.
COLORS = {
    "title":          "#dc2626",
    "section_header": "#ea580c",
    "figure":         "#2563eb",
    "table":          "#16a34a",
    "text":           "#6b7280",
}
DEFAULT_COLOR = "#9ca3af"

WIDTHS = {
    "title": 4,
    "section_header": 3,
    "figure": 3,
    "table": 3,
    "text": 2,
}
DEFAULT_WIDTH = 2


def color_for(elem_type: str) -> str:
    return COLORS.get(elem_type, DEFAULT_COLOR)


def width_for(elem_type: str) -> int:
    return WIDTHS.get(elem_type, DEFAULT_WIDTH)


def _hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


@st.cache_data(ttl=600, show_spinner=False)
def load_image_bytes(image_uri: str) -> bytes:
    resp = workspace_client().files.download(image_uri)
    return resp.contents.read()


def _load_label_font() -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, 18)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render_with_bboxes(image_bytes: bytes, elements: list[dict]) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_label_font()

    for el in elements:
        bbox = el.get("bbox") or []
        if not bbox or not isinstance(bbox[0], dict):
            continue
        coord = bbox[0].get("coord")
        if not coord or len(coord) < 4:
            continue

        x1, y1, x2, y2 = (int(coord[0]), int(coord[1]), int(coord[2]), int(coord[3]))
        elem_type = el.get("type", "") or ""
        color = color_for(elem_type)
        width = width_for(elem_type)

        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)

        label = f"#{el.get('element_id', '?')} {elem_type}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        pad = 4
        bg = _hex_to_rgba(color, alpha=200)
        draw.rectangle(
            (x1, y1, x1 + tw + pad * 2, y1 + th + pad * 2),
            fill=bg,
        )
        draw.text((x1 + pad, y1 + pad - 2), label, fill="white", font=font)

    composed = Image.alpha_composite(img, overlay)
    out = io.BytesIO()
    composed.convert("RGB").save(out, format="PNG", optimize=True)
    return out.getvalue()
