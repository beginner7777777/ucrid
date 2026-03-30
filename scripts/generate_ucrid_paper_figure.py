from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/mnt/data3/wzc/llm_oos_detection")
OUT_PATH = ROOT / "outputs" / "figures" / "ucrid_pipeline_diagram_20260321.png"


W, H = 2200, 1400
BG = "#FAFBFD"
TEXT = "#1F2937"
SUBTEXT = "#4B5563"
LINE = "#374151"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(44, bold=True)
FONT_H1 = load_font(34, bold=True)
FONT_H2 = load_font(26, bold=True)
FONT_BODY = load_font(23, bold=False)
FONT_SMALL = load_font(20, bold=False)


def draw_multiline_center(draw: ImageDraw.ImageDraw, box, text, font, fill=TEXT, spacing=6):
    x1, y1, x2, y2 = box
    lines = text.split("\n")
    line_heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_heights.append(bbox[3] - bbox[1])
    total_h = sum(line_heights) + spacing * (len(lines) - 1)
    y = y1 + (y2 - y1 - total_h) / 2
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = x1 + (x2 - x1 - w) / 2
        draw.text((x, y), line, font=font, fill=fill)
        y += h + spacing


def box(draw, xy, fill, outline=LINE, radius=26, width=3):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, p1, p2, fill=LINE, width=5, head=16):
    draw.line([p1, p2], fill=fill, width=width)
    x1, y1 = p1
    x2, y2 = p2
    if abs(x2 - x1) >= abs(y2 - y1):
        if x2 >= x1:
            pts = [(x2, y2), (x2 - head, y2 - head / 2), (x2 - head, y2 + head / 2)]
        else:
            pts = [(x2, y2), (x2 + head, y2 - head / 2), (x2 + head, y2 + head / 2)]
    else:
        if y2 >= y1:
            pts = [(x2, y2), (x2 - head / 2, y2 - head), (x2 + head / 2, y2 - head)]
        else:
            pts = [(x2, y2), (x2 - head / 2, y2 + head), (x2 + head / 2, y2 + head)]
    draw.polygon(pts, fill=fill)


def label(draw, xy, text, font=FONT_SMALL, fill=SUBTEXT):
    draw.text(xy, text, font=font, fill=fill)


def main():
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    draw.text((60, 35), "UCRID: Uncertainty-aware Cascade Routing for Intent and OOS Detection", font=FONT_TITLE, fill=TEXT)
    draw.text((62, 92), "Three-stage training and inference pipeline aligned with the implemented codebase", font=FONT_BODY, fill=SUBTEXT)

    # Section backgrounds
    train_bg = (40, 155, 2160, 690)
    infer_bg = (40, 735, 2160, 1325)
    draw.rounded_rectangle(train_bg, radius=36, fill="#F4F8FF", outline="#C7D2FE", width=3)
    draw.rounded_rectangle(infer_bg, radius=36, fill="#F9FAFB", outline="#D1D5DB", width=3)
    draw.text((70, 170), "Offline Training and Validation Calibration", font=FONT_H1, fill="#1D4ED8")
    draw.text((70, 750), "Online Inference and Cascade Decision", font=FONT_H1, fill="#111827")

    # Top row
    b_train = (90, 250, 330, 380)
    b_model = (410, 220, 810, 420)
    b_proto = (900, 235, 1220, 405)
    b_val = (1310, 235, 1640, 405)
    b_llmmeta = (1730, 220, 2085, 420)

    box(draw, b_train, fill="#FFFFFF")
    draw_multiline_center(draw, b_train, "Training Data\n(ID + optional OOS)", FONT_H2)

    box(draw, b_model, fill="#E8F1FF")
    draw_multiline_center(
        draw,
        b_model,
        "Stage 1 Small Model\nBERT encoder + MLP classifier\nLoss = CE + λs·SupCon + λb·Boundary",
        FONT_H2,
    )

    box(draw, b_proto, fill="#FEF3C7")
    draw_multiline_center(
        draw,
        b_proto,
        "Prototype Builder\nMean embedding per\nin-domain intent",
        FONT_H2,
    )

    box(draw, b_val, fill="#DCFCE7")
    draw_multiline_center(
        draw,
        b_val,
        "Validation Calibration\nTemperature scaling\nEntropy / distance normalization\nGrid search of routing thresholds",
        FONT_H2,
    )

    box(draw, b_llmmeta, fill="#FCE7F3")
    draw_multiline_center(
        draw,
        b_llmmeta,
        "Intent Metadata Builder\nintent names\nheuristic descriptions\nfew-shot pools\nOOS example pool",
        FONT_H2,
    )

    arrow(draw, (330, 315), (410, 315))
    arrow(draw, (810, 315), (900, 315))
    arrow(draw, (810, 340), (1310, 340))
    arrow(draw, (330, 285), (1730, 285))
    label(draw, (1010, 418), "prototype bank", FONT_SMALL)
    label(draw, (1435, 418), "router parameters", FONT_SMALL)
    label(draw, (1810, 428), "Stage 3 prompt resources", FONT_SMALL)

    # Divider
    draw.line((60, 710, 2140, 710), fill="#CBD5E1", width=4)

    # Inference row
    b_query = (90, 915, 320, 1035)
    b_s1 = (410, 850, 770, 1105)
    b_router = (860, 820, 1285, 1135)
    b_direct = (1380, 830, 1600, 940)
    b_oos = (1380, 1010, 1600, 1120)
    b_stage3 = (1690, 820, 2085, 1135)
    b_out = (1810, 1195, 2065, 1290)

    box(draw, b_query, fill="#FFFFFF")
    draw_multiline_center(draw, b_query, "Input Query u", FONT_H2)

    box(draw, b_s1, fill="#DBEAFE")
    draw_multiline_center(
        draw,
        b_s1,
        "Stage 1 Forward\nBERT encoding\nCLS embedding h(u)\nlogits z(u), top-1, top-k",
        FONT_H2,
    )

    box(draw, b_router, fill="#EDE9FE")
    draw_multiline_center(
        draw,
        b_router,
        "Stage 2 Router\nH(u) = entropy(softmax(z/T))\ndmin(u) = min ||h(u)-pc||2\ns(u) = α·Hnorm + (1-α)·dnorm\n\nIf s≤τaccept -> small model\nIf s≥τreject and dmin>Δ -> direct OOS\nElse -> Stage 3",
        FONT_H2,
    )

    box(draw, b_direct, fill="#D1FAE5")
    draw_multiline_center(draw, b_direct, "Accept\nsmall-model\nprediction", FONT_H2)

    box(draw, b_oos, fill="#FEE2E2")
    draw_multiline_center(draw, b_oos, "Direct OOS\nwithout LLM", FONT_H2)

    box(draw, b_stage3, fill="#FDE68A")
    draw_multiline_center(
        draw,
        b_stage3,
        "Stage 3 LLM Judge\nConstruct prompt from:\n1) top-k candidate intents\n2) intent definitions\n3) retrieved in-intent examples\n4) retrieved OOS examples\n\nParse label and apply policy:\nall / oos_only / id_only",
        FONT_H2,
    )

    box(draw, b_out, fill="#FBCFE8")
    draw_multiline_center(draw, b_out, "Final Intent / OOS", FONT_H2)

    arrow(draw, (320, 975), (410, 975))
    arrow(draw, (770, 975), (860, 975))
    arrow(draw, (1285, 900), (1380, 885))
    arrow(draw, (1285, 1070), (1380, 1065))
    arrow(draw, (1285, 975), (1690, 975))
    arrow(draw, (1888, 1135), (1935, 1195))
    arrow(draw, (1600, 885), (1810, 1230))
    arrow(draw, (1600, 1065), (1810, 1235))

    label(draw, (1322, 858), "s(u) ≤ τaccept", FONT_SMALL)
    label(draw, (1292, 1018), "s(u) ≥ τreject and dmin > Δ", FONT_SMALL)
    label(draw, (1448, 962), "otherwise", FONT_SMALL)

    # Cross-stage arrows from offline artifacts to online modules
    arrow(draw, (1060, 405), (1060, 820))
    arrow(draw, (1475, 405), (1180, 820))
    arrow(draw, (1900, 420), (1900, 820))
    label(draw, (1080, 575), "intent prototypes", FONT_SMALL)
    label(draw, (1220, 610), "T, α, τaccept, τreject, Δ", FONT_SMALL)
    label(draw, (1920, 610), "candidate descriptions\nand example pools", FONT_SMALL)

    # Bottom notes
    note_box = (90, 1185, 1705, 1305)
    box(draw, note_box, fill="#FFFFFF", outline="#CBD5E1", radius=24, width=2)
    draw_multiline_center(
        draw,
        note_box,
        "Implemented pipeline: train Stage 1 once, calibrate Stage 2 on validation set, then evaluate Stage 1 / Stage 1+2 / Stage 1+2+3 on the test set.\nResults are saved as ucrid_results.json and ucrid_prediction_details.json for downstream analysis and paper tables.",
        FONT_BODY,
    )

    img.save(OUT_PATH)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
