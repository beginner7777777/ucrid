from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
UCRID_DIR = ROOT / "outputs" / "ucrid"
FIG_PATH = ROOT / "outputs" / "figures" / "efficiency_tradeoff.png"
CSV_PATH = ROOT / "results" / "efficiency_summary.csv"


@dataclass
class ResultRow:
    dataset: str
    exp_name: str
    accuracy: Optional[float]
    id_accuracy: Optional[float]
    oos_f1: Optional[float]
    llm_call_rate: Optional[float]
    final_stage: str
    llm_enabled: Optional[bool]
    temp: Optional[float]
    tau_accept: Optional[float]
    tau_reject: Optional[float]
    delta: Optional[float]
    file_path: str


def infer_dataset(path_text: str) -> str:
    s = path_text.lower()
    if "clinc150" in s:
        return "clinc150"
    if "banking77" in s:
        return "banking77"
    if "stackoverflow" in s:
        return "stackoverflow"
    if "hint3" in s or "curekart" in s or "sofmattress" in s or "powerplay11" in s:
        return "hint3"
    return "unknown"


def read_results() -> List[ResultRow]:
    rows: List[ResultRow] = []
    for fp in sorted(UCRID_DIR.glob("**/ucrid_results.json")):
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue

        router_cfg = data.get("router_config", {})
        rows.append(
            ResultRow(
                dataset=infer_dataset(str(fp.parent)),
                exp_name=fp.parent.name,
                accuracy=data.get("accuracy"),
                id_accuracy=data.get("id_accuracy"),
                oos_f1=data.get("oos_f1"),
                llm_call_rate=data.get("llm_call_rate"),
                final_stage=data.get("final_stage", "unknown"),
                llm_enabled=data.get("llm_enabled"),
                temp=router_cfg.get("temperature"),
                tau_accept=router_cfg.get("tau_accept"),
                tau_reject=router_cfg.get("tau_reject"),
                delta=router_cfg.get("delta"),
                file_path=str(fp),
            )
        )
    return rows


def write_csv(rows: List[ResultRow]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "exp_name",
                "accuracy",
                "id_accuracy",
                "oos_f1",
                "llm_call_rate",
                "final_stage",
                "llm_enabled",
                "router_temperature",
                "tau_accept",
                "tau_reject",
                "delta",
                "file_path",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.dataset,
                    r.exp_name,
                    r.accuracy,
                    r.id_accuracy,
                    r.oos_f1,
                    r.llm_call_rate,
                    r.final_stage,
                    r.llm_enabled,
                    r.temp,
                    r.tau_accept,
                    r.tau_reject,
                    r.delta,
                    r.file_path,
                ]
            )


def method_label(exp_name: str) -> Optional[str]:
    s = exp_name.lower()
    if "wo_supcon" in s:
        return "w/o SupCon"
    if "wo_boundary" in s:
        return "w/o Boundary"
    if "single_threshold" in s:
        return "Single-threshold"
    if "full_loss_all_epochs" in s:
        return "Full-loss all-epochs"
    if "entropy_only" in s:
        return "Entropy-only"
    if "distance_only" in s:
        return "Distance-only"
    if "qwen3_8888_fixed" in s and "rerun" in s:
        return "Dual-threshold baseline"
    return None


def pick_latest(rows: List[ResultRow]) -> ResultRow:
    # rows come from sorted file paths; pick the latest by exp name lexical as a stable proxy.
    return sorted(rows, key=lambda r: r.exp_name)[-1]


def build_ablation_matrix(rows: List[ResultRow]) -> Dict[str, Dict[str, float]]:
    selected: Dict[str, Dict[str, List[ResultRow]]] = {
        "clinc150": {},
        "banking77": {},
    }
    for r in rows:
        if r.dataset not in selected:
            continue
        lbl = method_label(r.exp_name)
        if lbl is None or r.oos_f1 is None:
            continue
        selected[r.dataset].setdefault(lbl, []).append(r)

    out: Dict[str, Dict[str, float]] = {"clinc150": {}, "banking77": {}}
    for ds in out:
        for lbl, candidates in selected[ds].items():
            out[ds][lbl] = pick_latest(candidates).oos_f1
    return out


def plot(rows: List[ResultRow]) -> None:
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    colors = {
        "clinc150": "#1f77b4",
        "banking77": "#ff7f0e",
        "hint3": "#2ca02c",
        "stackoverflow": "#9467bd",
        "unknown": "#7f7f7f",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    # 1) Scatter: llm_call_rate vs oos_f1
    used_labels = set()
    for r in rows:
        if r.oos_f1 is None or r.llm_call_rate is None:
            continue
        c = colors.get(r.dataset, colors["unknown"])
        label = r.dataset if r.dataset not in used_labels else None
        ax1.scatter(r.llm_call_rate * 100.0, r.oos_f1 * 100.0, s=42, color=c, alpha=0.75, label=label)
        used_labels.add(r.dataset)

    ax1.set_title("Efficiency Trade-off: LLM Call Rate vs OOS F1")
    ax1.set_xlabel("LLM Call Rate (%)")
    ax1.set_ylabel("OOS F1 (%)")
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    if used_labels:
        ax1.legend(title="Dataset", frameon=True)

    # 2) Ablation bar chart
    ablations = build_ablation_matrix(rows)
    method_order = [
        "Dual-threshold baseline",
        "w/o SupCon",
        "w/o Boundary",
        "Single-threshold",
        "Entropy-only",
        "Distance-only",
        "Full-loss all-epochs",
    ]
    methods = [m for m in method_order if m in ablations["clinc150"] or m in ablations["banking77"]]

    x = list(range(len(methods)))
    width = 0.38
    clinc_vals = [ablations["clinc150"].get(m, float("nan")) * 100.0 for m in methods]
    bank_vals = [ablations["banking77"].get(m, float("nan")) * 100.0 for m in methods]

    ax2.bar([i - width / 2 for i in x], clinc_vals, width=width, color=colors["clinc150"], alpha=0.9, label="CLINC150")
    ax2.bar([i + width / 2 for i in x], bank_vals, width=width, color=colors["banking77"], alpha=0.9, label="Banking77")

    ax2.set_title("Ablation Comparison (OOS F1)")
    ax2.set_ylabel("OOS F1 (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=25, ha="right")
    ax2.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax2.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = read_results()
    write_csv(rows)
    plot(rows)
    print(f"Loaded result files: {len(rows)}")
    print(CSV_PATH)
    print(FIG_PATH)


if __name__ == "__main__":
    main()
