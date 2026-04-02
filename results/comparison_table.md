# Main Results: UCRID vs. Prior Work

> **Evaluation protocol note.** Numbers from prior literature (FCSLM, UDRIL, etc.) are reported under their original evaluation setups and may differ in known-intent split ratio, OOS injection strategy, or metric definition. Direct numerical comparison should be treated with caution. UCRID numbers are all from this project's unified evaluation pipeline (`run_ucrid.py`).

---

## 1. Main Comparison Table (CLINC150)

| Method | Venue | Acc. | ID Acc. | OOS Prec. | OOS Rec. | OOS F1 | LLM Rate |
|---|---|---:|---:|---:|---:|---:|---:|
| BERT MaxSoftmax (E1) | — | — | 95.51 | 93.80 | 61.30 | 73.55 | — |
| BERT + SupCon (E2) | — | — | 95.36 | 86.40 | 67.70 | 76.58 | — |
| BERT + SupCon + ProtoBank (E4) | — | — | 92.80 | 78.20 | **92.50** | 84.67 | — |
| Arora et al. | 2024 | — | — | — | — | ~82–85 | — |
| UDRIL | ACL 2025 | — | — | — | — | ~85–87 | — |
| FCSLM | EMNLP 2025 | ~93.8 | — | — | — | ~88.6 | — |
| UCRID + Qwen2-7B | ours | 90.51 | 95.96 | 94.29 | 66.00 | 77.65 | 8.1% |
| UCRID + Mixtral-8x7B | ours | 91.02 | 95.91 | 93.62 | 69.00 | 79.45 | 8.1% |
| UCRID + DeepSeek-R1-8B | ours | 89.60 | 95.93 | 93.28 | 61.10 | 73.84 | 8.1% |
| **UCRID + Qwen3-8B (best)** | ours | **92.40** | **95.91** | **93.76** | 76.60 | **84.31** | **8.1%** |

> `~` = approximate values from original papers; `—` = not reported under the current evaluation protocol.

---

## 2. Main Comparison Table (Banking77)

| Method | Venue | Acc. | ID Acc. | OOS Prec. | OOS Rec. | OOS F1 | LLM Rate |
|---|---|---:|---:|---:|---:|---:|---:|
| BERT baseline (Stage1 only) | — | 79.78 | **92.73** | 0.00 | 0.00 | 0.00 | — |
| UCRID Stage1+2 (no LLM) | ours | 89.94 | 92.47 | 95.38 | 74.40 | 83.60 | 7.2% |
| UCRID + DeepSeek-R1-8B | ours | 89.41 | 91.82 | 86.34 | 74.60 | 80.04 | 7.2% |
| UCRID + Mixtral-8x7B | ours | 91.76 | 92.47 | **95.00** | 87.40 | 91.04 | 7.2% |
| UCRID + Qwen2-7B | ours | 92.60 | 92.37 | 94.00 | 94.00 | 94.00 | 7.2% |
| **UCRID + Qwen3-8B (best)** | ours | **92.65** | 92.44 | 94.95 | **94.00** | **94.47** | **7.2%** |

---

## 3. LaTeX `tabular` — CLINC150

```latex
\begin{table}[t]
\centering
\caption{%
  Main results on CLINC150.
  Numbers from prior work (\dag) are reported under original evaluation setups and may not be directly comparable.
  UCRID numbers use the same unified evaluation pipeline.
  $\dagger$: approximate values from original papers.
}
\label{tab:main_clinc150}
\begin{tabular}{l l c c c c c c}
\hline
Method & Venue & Acc. & ID Acc. & OOS Prec. & OOS Rec. & OOS F1 & LLM Rate \\
\hline
BERT MaxSoftmax (E1)          & ---         & ---           & 95.51          & 93.80          & 61.30           & 73.55          & ---    \\
BERT + SupCon (E2)            & ---         & ---           & 95.36          & 86.40          & 67.70           & 76.58          & ---    \\
BERT + SupCon + Proto (E4)    & ---         & ---           & 92.80          & 78.20          & \textbf{92.50}  & 84.67          & ---    \\
Arora et al.$^\dagger$        & 2024        & ---           & ---            & ---            & ---             & \textasciitilde 82--85 & ---    \\
UDRIL$^\dagger$               & ACL 2025    & ---           & ---            & ---            & ---             & \textasciitilde 85--87 & ---    \\
FCSLM$^\dagger$               & EMNLP 2025  & \textasciitilde 93.8 & ---    & ---            & ---             & \textasciitilde 88.6   & ---    \\
\hline
UCRID + Qwen2-7B              & ours        & 90.51         & 95.96          & 94.29          & 66.00           & 77.65          & 8.1\%  \\
UCRID + Mixtral-8x7B          & ours        & 91.02         & 95.91          & 93.62          & 69.00           & 79.45          & 8.1\%  \\
UCRID + DeepSeek-R1-8B        & ours        & 89.60         & 95.93          & 93.28          & 61.10           & 73.84          & 8.1\%  \\
\textbf{UCRID + Qwen3-8B}     & ours        & \textbf{92.40}& \textbf{95.91} & \textbf{93.76} & 76.60           & \textbf{84.31} & \textbf{8.1\%} \\
\hline
\end{tabular}
\end{table}
```

---

## 4. LaTeX `tabular` — Banking77

```latex
\begin{table}[t]
\centering
\caption{%
  Main results on Banking77 (OOS samples injected from CLINC150 OOS pool).
  All UCRID variants share the same Stage-1 backbone and Stage-2 router;
  only the Stage-3 LLM backend differs.
}
\label{tab:main_banking77}
\begin{tabular}{l l c c c c c c}
\hline
Method & Venue & Acc. & ID Acc. & OOS Prec. & OOS Rec. & OOS F1 & LLM Rate \\
\hline
Stage1 only (BERT)            & ---   & 79.78          & \textbf{92.73} & 0.00           & 0.00           & 0.00           & ---    \\
Stage1+2 (no LLM)             & ours  & 89.94          & 92.47          & 95.38          & 74.40          & 83.60          & 7.2\%  \\
\hline
UCRID + DeepSeek-R1-8B        & ours  & 89.41          & 91.82          & 86.34          & 74.60          & 80.04          & 7.2\%  \\
UCRID + Mixtral-8x7B          & ours  & 91.76          & 92.47          & \textbf{95.00} & 87.40          & 91.04          & 7.2\%  \\
UCRID + Qwen2-7B              & ours  & 92.60          & 92.37          & 94.00          & 94.00          & 94.00          & 7.2\%  \\
\textbf{UCRID + Qwen3-8B}     & ours  & \textbf{92.65} & 92.44          & 94.95          & \textbf{94.00} & \textbf{94.47} & \textbf{7.2\%} \\
\hline
\end{tabular}
\end{table}
```

---

## 5. Analysis: UCRID vs. FCSLM / UDRIL

### 5.1 Advantages of UCRID

**Low and controllable LLM usage.**
UCRID invokes the LLM for only 8.1% of CLINC150 queries and 7.2% of Banking77 queries to reach its best OOS F1. FCSLM and UDRIL are end-to-end fine-tuned models that process every sample through a full-scale pipeline. UCRID's cascade architecture makes LLM cost predictable and adjustable at deployment time by tuning the dual-threshold router.

**Strong cross-dataset generalization on Banking77.**
UCRID achieves 94.47 OOS F1 on Banking77, a fine-grained 77-class banking domain where FCSLM and UDRIL have not been reported. This suggests that the cascade design — prototype-distance routing plus constrained LLM arbitration — generalizes effectively to fine-grained, domain-specific intent boundaries, not only to the standard multi-domain CLINC150 benchmark.

**Interpretable and modular routing.**
Each component of UCRID has a clear role: the backbone encoder produces representations, the dual-signal router separates easy from ambiguous samples, and the LLM adjudicates hard cases. Each stage is independently ablatable (see ablation tables), which makes it easier to diagnose failure modes and tune behavior in production. FCSLM and UDRIL, as end-to-end systems, offer less transparency into which samples drive performance.

**High OOS recall with surgical LLM use.**
On CLINC150, UCRID's best result achieves 76.60 OOS recall with the LLM arbitrating only 8.1% of samples. Across both datasets, this shows that the cascade effectively concentrates LLM computation on genuinely ambiguous cases rather than applying high-capacity reasoning uniformly.

---

### 5.2 Disadvantages of UCRID

**CLINC150 OOS F1 gap relative to FCSLM.**
FCSLM reports approximately 88.6 OOS F1 on CLINC150 against UCRID's 84.31. The gap is roughly 4 OOS F1 points. This matters because CLINC150 is the standard benchmark. UCRID partially compensates with better interpretability and lower inference cost, but the single-number comparison on this benchmark is not in UCRID's favor.

**Sensitivity to router threshold calibration.**
Unlike end-to-end fine-tuned baselines, UCRID requires validation-time search over four router hyperparameters (α, τ_accept, τ_reject, Δ) and temperature scaling. This adds engineering overhead and may not transfer reliably to new datasets without re-calibration. The HINT3-v2 results illustrate this: UCRID reaches strong OOS recall but weaker in-domain accuracy under cross-domain shift, partly because the router was not re-calibrated for the HINT3 distribution.

**CLINC150 OOS recall does not reach E4's level.**
An internal comparison reveals that the simpler E4 prototype-bank baseline achieves 92.50 OOS recall on CLINC150 under a Stage1+2 protocol, while full UCRID (Stage1+2+3) reaches 76.60 recall. The discrepancy arises from pipeline differences — E4 used an offline evaluation protocol while UCRID uses a real cascade with router constraints. This complicates a clean internal narrative and requires careful methodological exposition in the paper.

**LLM backend dependency.**
UCRID's performance at Stage 3 is sensitive to the LLM backend. DeepSeek-R1-Distill-Llama-8B, despite being a reasoning-augmented model, underperforms Qwen3-8B on both datasets within UCRID. Mixtral-8x7B is also slower by more than one order of magnitude in P95 latency. This means the "low LLM rate" advantage is only guaranteed if a suitable backend is available. In deployment contexts without a capable local LLM, the cascade advantage may be undermined.

---

### 5.3 Summary

| Criterion | UCRID (ours) | FCSLM / UDRIL |
|---|---|---|
| CLINC150 OOS F1 | 84.31 (gap ~4 pts vs. FCSLM) | ~88.6 / ~85–87 |
| Banking77 OOS F1 | **94.47** | Not reported |
| LLM call rate | **8.1% / 7.2%** | Full pipeline (100%) |
| Routing interpretability | **High** (ablatable stages) | Low (end-to-end) |
| Threshold calibration overhead | Required | Not required |
| Cross-domain sensitivity | Moderate (HINT3 degradation) | Not reported |

**Conclusion.** If the goal is maximizing CLINC150 OOS F1 on a single-benchmark leaderboard, FCSLM and UDRIL are stronger. If the goal is a deployable system with low, controllable LLM cost, interpretable routing, and demonstrated cross-dataset performance, UCRID provides a more practical and better-characterized solution.

---
