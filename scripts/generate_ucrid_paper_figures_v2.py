from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/mnt/data3/wzc/llm_oos_detection")
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DUAL_PATH = FIG_DIR / "ucrid_dual_column_train_infer_20260321.png"
FLOW_PATH = FIG_DIR / "ucrid_algorithm_flowchart_20260321.png"

TEXT = "#1F2937"
SUB = "#4B5563"
LINE = "#334155"
BG = "#FCFCFD"


def font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


FT = font(42, True)
FH1 = font(30, True)
FH2 = font(24, True)
FB = font(21, False)
FS = font(18, False)


def box(draw, xy, fill, outline=LINE, width=3, radius=24):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def center_text(draw, xy, text, fnt, fill=TEXT, spacing=5):
    x1, y1, x2, y2 = xy
    lines = text.split("\n")
    sizes = [draw.textbbox((0, 0), line, font=fnt) for line in lines]
    heights = [b[3] - b[1] for b in sizes]
    total_h = sum(heights) + spacing * (len(lines) - 1)
    y = y1 + (y2 - y1 - total_h) / 2
    for line, bbox, h in zip(lines, sizes, heights):
        w = bbox[2] - bbox[0]
        x = x1 + (x2 - x1 - w) / 2
        draw.text((x, y), line, font=fnt, fill=fill)
        y += h + spacing


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


def note(draw, xy, text, fnt=FS, fill=SUB):
    draw.text(xy, text, font=fnt, fill=fill)


def make_dual_column():
    w, h = 2200, 1350
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)

    draw.text((60, 35), "UCRID Training and Inference Overview", font=FT, fill=TEXT)
    draw.text((62, 90), "Paper-style dual-column view of the implemented pipeline", font=FB, fill=SUB)

    left = (50, 155, 1060, 1270)
    right = (1140, 155, 2150, 1270)
    draw.rounded_rectangle(left, radius=34, fill="#F5F9FF", outline="#BFDBFE", width=3)
    draw.rounded_rectangle(right, radius=34, fill="#FFFDF6", outline="#FCD34D", width=3)

    draw.text((85, 180), "Training Stage", font=FH1, fill="#1D4ED8")
    draw.text((1175, 180), "Inference Stage", font=FH1, fill="#B45309")

    # Left column
    l1 = (95, 255, 445, 375)
    l2 = (555, 245, 965, 385)
    l3 = (555, 445, 965, 595)
    l4 = (555, 655, 965, 825)
    l5 = (555, 925, 965, 1085)
    l6 = (95, 1080, 965, 1215)

    box(draw, l1, "#FFFFFF")
    center_text(draw, l1, "Training Set\n(ID samples + optional OOS samples)", FH2)

    box(draw, l2, "#E0ECFF")
    center_text(draw, l2, "Phase 1\nCE-only training\nlearn basic intent classification", FH2)

    box(draw, l3, "#DBEAFE")
    center_text(draw, l3, "Phase 2\nCE + SupCon\ncompact intent clusters", FH2)

    box(draw, l4, "#D1FAE5")
    center_text(draw, l4, "Phase 3\nCE + SupCon + Boundary\npush OOS away from ID prototypes", FH2)

    box(draw, l5, "#FEF3C7")
    center_text(draw, l5, "Validation Calibration\nfit temperature T\nnormalize entropy and distance\nsearch α, τaccept, τreject, Δ", FH2)

    box(draw, l6, "#FCE7F3")
    center_text(draw, l6, "Offline Artifacts\nbest_model.pt\nintent prototypes\nrouter parameters\nintent descriptions and example pools", FH2)

    arrow(draw, (445, 315), (555, 315))
    arrow(draw, (760, 385), (760, 445))
    arrow(draw, (760, 595), (760, 655))
    arrow(draw, (760, 825), (760, 925))
    arrow(draw, (760, 1085), (760, 1080))

    note(draw, (120, 405), "Three-phase schedule from train.py", FS)
    note(draw, (120, 435), "Loss from multi_task_loss.py", FS)

    # Right column
    r1 = (1185, 255, 1465, 375)
    r2 = (1550, 245, 2040, 405)
    r3 = (1550, 495, 2040, 755)
    r4a = (1550, 855, 1780, 965)
    r4b = (1810, 855, 2040, 965)
    r5 = (1550, 1030, 2040, 1195)

    box(draw, r1, "#FFFFFF")
    center_text(draw, r1, "Input Query u", FH2)

    box(draw, r2, "#E0ECFF")
    center_text(draw, r2, "Stage 1 Small Model\nBERT forward pass\noutput logits z(u)\nand embedding h(u)", FH2)

    box(draw, r3, "#EDE9FE")
    center_text(draw, r3, "Stage 2 Router\nentropy H(u)\nprototype distance dmin(u)\nuncertainty score s(u)\n\ns = α·Hnorm + (1-α)·dnorm\n\nsmall_model / direct_oos / llm", FH2)

    box(draw, r4a, "#D1FAE5")
    center_text(draw, r4a, "Fast Path\naccept top-1\nor direct OOS", FH2)

    box(draw, r4b, "#FDE68A")
    center_text(draw, r4b, "Stage 3 LLM\ncandidate-constrained\nfew-shot judgment", FH2)

    box(draw, r5, "#FBCFE8")
    center_text(draw, r5, "Prediction Fusion\napply all / oos_only / id_only\nsave final metrics and details", FH2)

    arrow(draw, (1465, 315), (1550, 315))
    arrow(draw, (1795, 405), (1795, 495))
    arrow(draw, (1715, 755), (1665, 855))
    arrow(draw, (1870, 755), (1925, 855))
    arrow(draw, (1665, 965), (1700, 1030))
    arrow(draw, (1925, 965), (1890, 1030))

    note(draw, (1220, 430), "run_ucrid.py evaluates Stage1, Stage1+2, Stage1+2+3", FS)
    note(draw, (1220, 460), "llm_judge.py builds prompt and parses label", FS)

    img.save(DUAL_PATH)


def make_flowchart():
    w, h = 1600, 1750
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)

    draw.text((50, 35), "UCRID Algorithm Flowchart", font=FT, fill=TEXT)
    draw.text((52, 90), "Step-by-step decision path for one input query", font=FB, fill=SUB)

    nodes = [
        ((480, 170, 1120, 280), "#FFFFFF", "Input query u"),
        ((390, 340, 1210, 490), "#DBEAFE", "Stage 1: Run BERT encoder\nGet logits z(u), embedding h(u), top-k intents"),
        ((390, 560, 1210, 760), "#EDE9FE", "Stage 2: Compute uncertainty\n1) temperature-scaled entropy H(u)\n2) nearest-prototype distance dmin(u)\n3) fused score s(u) = α·Hnorm + (1-α)·dnorm"),
        ((220, 850, 620, 980), "#D1FAE5", "s(u) ≤ τaccept ?\nYes -> accept small-model top-1"),
        ((680, 850, 1380, 980), "#FEE2E2", "Else, check\ns(u) ≥ τreject and dmin(u) > Δ ?\nYes -> direct OOS"),
        ((470, 1070, 1130, 1285), "#FDE68A", "Otherwise -> Stage 3 LLM\nBuild prompt with top-k intents,\nintent descriptions, in-intent examples,\nand OOS examples"),
        ((470, 1360, 1130, 1495), "#FEF3C7", "Parse LLM label\nnormalize output\nmap to intent id / OOS id"),
        ((470, 1560, 1130, 1690), "#FBCFE8", "Apply accept policy\nall / oos_only / id_only\nReturn final label"),
    ]

    for xy, fill, text in nodes:
        box(draw, xy, fill)
        center_text(draw, xy, text, FH2)

    arrow(draw, (800, 280), (800, 340))
    arrow(draw, (800, 490), (800, 560))
    arrow(draw, (800, 760), (420, 850))
    arrow(draw, (800, 760), (1030, 850))
    arrow(draw, (1030, 980), (800, 1070))
    arrow(draw, (800, 1285), (800, 1360))
    arrow(draw, (800, 1495), (800, 1560))

    note(draw, (245, 995), "No further LLM call", FS)
    note(draw, (885, 995), "No further LLM call", FS)
    note(draw, (850, 810), "If neither fast condition is satisfied, query enters the ambiguous region", FS)

    # side brackets / comments
    draw.rounded_rectangle((70, 320, 290, 760), radius=24, outline="#93C5FD", width=3, fill="#F8FBFF")
    center_text(draw, (70, 320, 290, 760), "Feature\nExtraction\n+\nRouting\nRegion", FH2, fill="#1D4ED8")

    draw.rounded_rectangle((1295, 1045, 1520, 1690), radius=24, outline="#F59E0B", width=3, fill="#FFF8EB")
    center_text(draw, (1295, 1045, 1520, 1690), "LLM\nJudgment\n+\nPolicy\nFusion", FH2, fill="#B45309")

    img.save(FLOW_PATH)


def main():
    make_dual_column()
    make_flowchart()
    print(DUAL_PATH)
    print(FLOW_PATH)


if __name__ == "__main__":
    main()
