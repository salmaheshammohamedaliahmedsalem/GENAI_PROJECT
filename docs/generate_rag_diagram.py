"""Generate a PNG diagram of the Mix RAG pipeline using Pillow."""
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "rag_pipeline.png"

W, H = 1500, 1080

# Palette
C = {
    "bg":           "#f8fafc",
    "blue_f":       "#dbeafe", "blue_b":   "#3b82f6",
    "purple_f":     "#ede9fe", "purple_b": "#7c3aed",
    "yellow_f":     "#fef9c3", "yellow_b": "#ca8a04",
    "green_f":      "#dcfce7", "green_b":  "#16a34a",
    "indigo_f":     "#e0f2fe", "indigo_b": "#0284c7",
    "orange_f":     "#fef3c7", "orange_b": "#d97706",
    "red_f":        "#ffe4e6", "red_b":    "#e11d48",
    "pink_f":       "#fce7f3", "pink_b":   "#db2777",
    "offclus_f":    "#f0fdf4", "offclus_b": "#86efac",
    "onclus_f":     "#eef2ff", "onclus_b":  "#a5b4fc",
    "arrow":        "#1e293b",
    "text":         "#000000",
    "muted":        "#1e293b",
}

img = Image.new("RGB", (W, H), C["bg"])
d   = ImageDraw.Draw(img)

# ── fonts ────────────────────────────────────────────────────────────────────
def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    paths = [
        f"/usr/share/fonts/truetype/dejavu/{name}.ttf",
        f"/usr/share/fonts/truetype/liberation/Liberation{name.replace('DejaVuSans','Sans')}.ttf",
        f"/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()

F_TITLE  = _font("DejaVuSans-Bold", 16)
F_BODY   = _font("DejaVuSans",      14)
F_SMALL  = _font("DejaVuSans",      12)
F_LABEL  = _font("DejaVuSans-Bold", 12)
F_CLUS   = _font("DejaVuSans-Bold", 13)
F_EDGE   = _font("DejaVuSans",      11)

# ── primitives ───────────────────────────────────────────────────────────────
def rrect(cx, cy, w, h, fill, border, r=12, bw=2):
    x0, y0 = cx - w // 2, cy - h // 2
    d.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=r, fill=fill, outline=border, width=bw)
    return x0, y0, x0 + w, y0 + h

def oval(cx, cy, w, h, fill, border, bw=2):
    x0, y0 = cx - w // 2, cy - h // 2
    d.ellipse([x0, y0, x0 + w, y0 + h], fill=fill, outline=border, width=bw)
    return x0, y0, x0 + w, y0 + h

def diamond(cx, cy, w, h, fill, border, bw=2):
    pts = [(cx, cy - h // 2), (cx + w // 2, cy), (cx, cy + h // 2), (cx - w // 2, cy)]
    d.polygon(pts, fill=fill)
    for i in range(len(pts)):
        d.line([pts[i], pts[(i + 1) % len(pts)]], fill=border, width=bw)
    return pts

def text_center(cx, cy, lines, fonts, color=None):
    color = color or C["text"]
    if isinstance(lines, str):
        lines = [lines]
        fonts = [fonts]
    lh = 18
    total = len(lines) * lh
    y = cy - total // 2 + lh // 2
    for line, fnt in zip(lines, fonts):
        d.text((cx, y), line, font=fnt, fill=color, anchor="mm")
        y += lh

def arrow(x1, y1, x2, y2, color=None, w=2, ah=10, aa=0.38):
    color = color or C["arrow"]
    d.line([x1, y1, x2, y2], fill=color, width=w)
    ang = math.atan2(y2 - y1, x2 - x1)
    for sign in (-aa, aa):
        ax = x2 - ah * math.cos(ang - sign)
        ay = y2 - ah * math.sin(ang - sign)
        d.line([(x2, y2), (int(ax), int(ay))], fill=color, width=w)

def dashed(x1, y1, x2, y2, color, w=2, dl=8):
    dx, dy = x2 - x1, y2 - y1
    ln = math.hypot(dx, dy)
    if ln == 0:
        return
    ux, uy = dx / ln, dy / ln
    pos, on = 0.0, True
    while pos < ln:
        nxt = min(pos + dl, ln)
        if on:
            d.line([(int(x1 + ux * pos), int(y1 + uy * pos)),
                    (int(x1 + ux * nxt), int(y1 + uy * nxt))], fill=color, width=w)
        pos = nxt + dl
        on = not on

def edge_label(cx, cy, txt, color=None):
    d.text((cx, cy), txt, font=F_EDGE, fill=color or C["muted"], anchor="mm")

# ── layout constants ─────────────────────────────────────────────────────────
CX = W // 2          # 750

Y_QUERY  = 70
Y_EXPAND = 160
Y_ROUTER = 268

# Cluster vertical span
CLUS_TOP    = 340
CLUS_BOTTOM = 590

OFF_CX = 290         # offline cluster centre-x
ON_CX  = 1095        # online cluster centre-x

Y_MERGE  = 680
Y_RERANK = 790
Y_CE     = 790       # cross-encoder (same row, right)
Y_RESULT = 920

# ── 1. User Query ────────────────────────────────────────────────────────────
oval(CX, Y_QUERY, 210, 48, C["blue_f"], C["blue_b"], bw=3)
text_center(CX, Y_QUERY, "User Query", F_TITLE)

arrow(CX, Y_QUERY + 24, CX, Y_EXPAND - 28)

# ── 2. Query Expansion ───────────────────────────────────────────────────────
rrect(CX, Y_EXPAND, 310, 56, C["purple_f"], C["purple_b"], r=14, bw=2)
text_center(CX, Y_EXPAND,
            ["Query Expansion  (enriched_query)", "clean · deduplicate · domain synonyms"],
            [F_LABEL, F_SMALL])

arrow(CX, Y_EXPAND + 28, CX, Y_ROUTER - 34)

# ── 3. Mode Router ───────────────────────────────────────────────────────────
diamond(CX, Y_ROUTER, 220, 68, C["yellow_f"], C["yellow_b"])
text_center(CX, Y_ROUTER, "Mode Router", F_LABEL)

# arrows from router to clusters
#  → offline
arrow(CX - 110, Y_ROUTER, OFF_CX + 115, CLUS_TOP + 22, w=2)
edge_label((CX - 110 + OFF_CX + 115) // 2 - 40,
           (Y_ROUTER + CLUS_TOP + 22) // 2 - 10,
           "offline / hybrid")

#  → online (to top of online cluster)
arrow(CX + 110, Y_ROUTER, ON_CX - 165, CLUS_TOP + 22, w=2)
edge_label((CX + 110 + ON_CX - 165) // 2 + 40,
           (Y_ROUTER + CLUS_TOP + 22) // 2 - 10,
           "online / hybrid")

# ── 4. Offline cluster ───────────────────────────────────────────────────────
OFF_L, OFF_R = OFF_CX - 130, OFF_CX + 130
d.rounded_rectangle([OFF_L, CLUS_TOP, OFF_R, CLUS_BOTTOM],
                    radius=16, fill=C["offclus_f"], outline=C["offclus_b"], width=2)
d.text((OFF_CX, CLUS_TOP + 16), "Offline Retrieval", font=F_CLUS, fill="#000000", anchor="mm")

BM25_Y   = CLUS_TOP + 100
CHROMA_Y = CLUS_TOP + 185

rrect(OFF_CX, BM25_Y,   200, 44, C["green_f"], C["green_b"], r=10)
text_center(OFF_CX, BM25_Y,   ["BM25", "(keyword search)"],   [F_LABEL, F_SMALL])

rrect(OFF_CX, CHROMA_Y, 200, 44, C["green_f"], C["green_b"], r=10)
text_center(OFF_CX, CHROMA_Y, ["ChromaDB", "(semantic search)"], [F_LABEL, F_SMALL])

# arrows offline → merge
arrow(OFF_CX, CLUS_BOTTOM, OFF_CX, Y_MERGE - 22, w=2)

# ── 5. Online cluster ────────────────────────────────────────────────────────
ON_L, ON_R = ON_CX - 185, ON_CX + 185
d.rounded_rectangle([ON_L, CLUS_TOP, ON_R, CLUS_BOTTOM],
                    radius=16, fill=C["onclus_f"], outline=C["onclus_b"], width=2)
d.text((ON_CX, CLUS_TOP + 16), "MultiSourceOnlineRetriever  (7 providers)",
       font=F_CLUS, fill="#000000", anchor="mm")

providers = [
    ("DuckDuckGo",       0, 0),
    ("Wikipedia",        1, 0),
    ("arXiv",            0, 1),
    ("Semantic Scholar", 1, 1),
    ("GitHub",           0, 2),
    ("StackExchange",    1, 2),
    ("YouTube *",        0, 3),
]
P_W, P_H = 158, 38
P_COL_OFF = 96   # x offset from cluster centre for each column
P_ROW_START = CLUS_TOP + 52
P_ROW_GAP   = 54

for label, col, row in providers:
    px = ON_CX - P_COL_OFF + col * (P_COL_OFF * 2)
    py = P_ROW_START + row * P_ROW_GAP
    rrect(px, py, P_W, P_H, C["indigo_f"], C["indigo_b"], r=8, bw=1)
    text_center(px, py, label, F_SMALL)

# arrow online cluster → merge
arrow(ON_CX, CLUS_BOTTOM, ON_CX, Y_MERGE - 22, w=2)

# ── 6. Merge Candidates ──────────────────────────────────────────────────────
rrect(CX, Y_MERGE, 300, 48, C["orange_f"], C["orange_b"], r=12, bw=2)
text_center(CX, Y_MERGE, "Merge Candidates", F_LABEL)

# curved merge arrows from offline & online clusters toward merge box
# (already drawn straight above; add small elbow lines)
arrow(OFF_CX, Y_MERGE, CX - 150, Y_MERGE, w=2)
arrow(ON_CX,  Y_MERGE, CX + 150, Y_MERGE, w=2)

arrow(CX, Y_MERGE + 24, CX, Y_RERANK - 28, w=2)

# ── 7. Cross-Encoder (optional, dashed box) ──────────────────────────────────
CE_CX = CX + 430
rrect(CE_CX, Y_CE, 230, 52, C["pink_f"], C["pink_b"], r=10, bw=1)
text_center(CE_CX, Y_CE,
            ["Cross-Encoder", "(optional — sentence-transformers)"],
            [F_LABEL, F_SMALL])

# dashed arrow from cross-encoder to reranker
dashed(CE_CX - 115, Y_CE, CX + 150, Y_RERANK, C["pink_b"], w=2)
edge_label((CE_CX - 115 + CX + 150) // 2 + 10, Y_CE - 14, "optional")

# ── 8. 4-Signal Reranker ─────────────────────────────────────────────────────
rrect(CX, Y_RERANK, 420, 58, C["red_f"], C["red_b"], r=14, bw=2)
text_center(CX, Y_RERANK,
            ["4-Signal Reranker",
             "Relevance  ·  Keyword  ·  Metadata  ·  Authority"],
            [F_LABEL, F_SMALL])

arrow(CX, Y_RERANK + 29, CX, Y_RESULT - 26, w=2)

# ── 9. Final Results ─────────────────────────────────────────────────────────
oval(CX, Y_RESULT, 270, 52, C["green_f"], C["green_b"], bw=3)
text_center(CX, Y_RESULT,
            ["RetrievedChunk List  (ranked, top-k)"],
            [F_TITLE])

# ── legend ───────────────────────────────────────────────────────────────────
LX, LY = 30, H - 90
d.text((LX, LY), "* YouTube requires YOUTUBE_API_KEY", font=F_SMALL, fill=C["muted"])
d.text((LX, LY + 20), "Cross-encoder reranking is optional (sentence-transformers).", font=F_SMALL, fill=C["muted"])
d.text((LX, LY + 40), "Offline: reads project BM25 index + ChromaDB.  Online: live web, no API key required for Tier 1.", font=F_SMALL, fill=C["muted"])

# ── title ────────────────────────────────────────────────────────────────────
d.text((CX, 20), "Mix RAG Pipeline", font=_font("DejaVuSans-Bold", 20), fill=C["text"], anchor="mm")

img.save(OUT, dpi=(150, 150))
print(f"Saved → {OUT}")
