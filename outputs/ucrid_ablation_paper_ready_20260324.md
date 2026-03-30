# UCRID Ablation Study for Paper Writing

## 1. Ablation Setting

This ablation study is conducted on the current `run_ucrid.py` pipeline under the best-performing `Qwen3-8B fixed + oos_only` setting. The goal is to isolate the contribution of each major component in UCRID:

- `Stage1 only`: the backbone intent classifier only
- `Stage1+2 (full router)`: uncertainty-aware routing without Stage-3 LLM arbitration
- `w/o distance signal (entropy-only)`: Stage-2 router uses only entropy (`alpha=1.0`)
- `w/o entropy signal (distance-only)`: Stage-2 router uses only prototype distance (`alpha=0.0`)
- `Full UCRID`: Stage1 + Stage2 + Stage3

The ablation is evaluated on both CLINC150 and Banking77 using the same metric suite: overall accuracy, ID accuracy, OOS precision, OOS recall, OOS F1, and LLM call rate.

## 2. Paper-Ready Tables

### 2.1 CLINC150

| Method | Acc. | ID Acc. | OOS Prec. | OOS Rec. | OOS F1 | LLM Rate |
|---|---:|---:|---:|---:|---:|---:|
| Stage1 only | 86.35 | 96.07 | 95.73 | 42.60 | 58.96 | - |
| Stage1+2 (full router) | 89.55 | 95.96 | 93.82 | 60.70 | 73.71 | 8.1 |
| w/o distance (entropy-only) | 89.11 | 95.98 | 94.33 | 58.20 | 71.99 | 5.9 |
| w/o entropy (distance-only) | 91.07 | 95.76 | 91.50 | 70.00 | 79.32 | 9.6 |
| Full UCRID | **92.40** | 95.91 | 93.76 | **76.60** | **84.31** | 8.1 |

### 2.2 Banking77

| Method | Acc. | ID Acc. | OOS Prec. | OOS Rec. | OOS F1 | LLM Rate |
|---|---:|---:|---:|---:|---:|---:|
| Stage1 only | 79.78 | **92.73** | 0.00 | 0.00 | 0.00 | - |
| Stage1+2 (full router) | 89.94 | 92.47 | 95.38 | 74.40 | 83.60 | 7.2 |
| w/o distance (entropy-only) | 89.61 | 92.53 | **95.72** | 71.60 | 81.92 | 6.2 |
| w/o entropy (distance-only) | 91.17 | 92.34 | 91.50 | 84.00 | 87.59 | 11.9 |
| Full UCRID | **92.65** | 92.44 | 94.95 | **94.00** | **94.47** | 7.2 |

## 3. LaTeX Tables

### 3.1 CLINC150 LaTeX

```tex
\begin{table}[t]
\centering
\caption{Ablation results on CLINC150.}
\label{tab:ablation_clinc150}
\begin{tabular}{lcccccc}
\hline
Method & Acc. & ID Acc. & OOS Prec. & OOS Rec. & OOS F1 & LLM Rate \\
\hline
Stage1 only & 86.35 & 96.07 & 95.73 & 42.60 & 58.96 & - \\
Stage1+2 (full router) & 89.55 & 95.96 & 93.82 & 60.70 & 73.71 & 8.1 \\
w/o distance (entropy-only) & 89.11 & 95.98 & 94.33 & 58.20 & 71.99 & 5.9 \\
w/o entropy (distance-only) & 91.07 & 95.76 & 91.50 & 70.00 & 79.32 & 9.6 \\
Full UCRID & \textbf{92.40} & 95.91 & 93.76 & \textbf{76.60} & \textbf{84.31} & 8.1 \\
\hline
\end{tabular}
\end{table}
```

### 3.2 Banking77 LaTeX

```tex
\begin{table}[t]
\centering
\caption{Ablation results on Banking77.}
\label{tab:ablation_banking77}
\begin{tabular}{lcccccc}
\hline
Method & Acc. & ID Acc. & OOS Prec. & OOS Rec. & OOS F1 & LLM Rate \\
\hline
Stage1 only & 79.78 & \textbf{92.73} & 0.00 & 0.00 & 0.00 & - \\
Stage1+2 (full router) & 89.94 & 92.47 & 95.38 & 74.40 & 83.60 & 7.2 \\
w/o distance (entropy-only) & 89.61 & 92.53 & \textbf{95.72} & 71.60 & 81.92 & 6.2 \\
w/o entropy (distance-only) & 91.17 & 92.34 & 91.50 & 84.00 & 87.59 & 11.9 \\
Full UCRID & \textbf{92.65} & 92.44 & 94.95 & \textbf{94.00} & \textbf{94.47} & 7.2 \\
\hline
\end{tabular}
\end{table}
```

## 4. English Analysis Paragraphs

### 4.1 Main Ablation Analysis

The ablation results consistently demonstrate that Stage-2 routing is indispensable for effective OOS detection. On CLINC150, introducing the router improves OOS F1 from 58.96 to 73.71. On Banking77, the gain is even larger, where OOS F1 rises from 0.00 to 83.60. This shows that the backbone classifier alone is insufficient for reliable OOS recognition, while uncertainty-aware routing provides the first substantial performance jump.

Among the two routing signals, prototype distance is clearly more informative than entropy. On CLINC150, removing entropy while retaining distance yields an OOS F1 of 79.32, compared with 71.99 for the entropy-only variant. A similar pattern is observed on Banking77, where the distance-only variant reaches 87.59 OOS F1, surpassing the entropy-only result of 81.92. These results suggest that OOS samples are more effectively characterized as instances lying far from the known intent prototypes, whereas entropy alone cannot fully capture this structural deviation.

However, the best Stage-2 variant is still not the final optimum. Full UCRID further improves OOS F1 to 84.31 on CLINC150 and 94.47 on Banking77, indicating that Stage-3 LLM arbitration remains essential. This confirms that the router and the LLM play complementary roles: the router identifies uncertain regions efficiently, and the LLM resolves fine-grained ambiguity within those hard cases.

### 4.2 Interpretation of the Dual-Signal Router

An additional observation is that the full dual-signal router does not always outperform the distance-only variant at the Stage-2 level. This does not contradict the design goal of UCRID. In the current pipeline, the router is selected through validation-based search under both performance and call-budget considerations, rather than by maximizing Stage-2 OOS F1 alone. As a result, the fused router serves as a better trade-off mechanism between detection quality and LLM usage, while the final performance ceiling is achieved after Stage-3 arbitration.

### 4.3 Short Version for the Paper

Table~\ref{tab:ablation_clinc150} and Table~\ref{tab:ablation_banking77} present the ablation results on CLINC150 and Banking77. First, Stage-2 routing is necessary for practical OOS detection, as it substantially improves OOS F1 over the backbone classifier alone on both datasets. Second, prototype distance contributes more than entropy as the dominant routing signal, since the distance-only variant consistently outperforms the entropy-only variant. Third, the best overall results are achieved only when the router is combined with Stage-3 LLM arbitration, showing that uncertainty-aware routing and candidate-constrained LLM judgment are complementary rather than redundant.

## 5. One-Sentence Conclusion

The ablation study shows that UCRID gains its effectiveness from three progressively stronger components: Stage-2 routing enables OOS detection, prototype distance provides the strongest routing signal, and Stage-3 LLM arbitration delivers the final performance gains.

## 6. Source Files

- Raw ablation summary: [ucrid_ablation_results_20260324.md](/mnt/data3/wzc/llm_oos_detection/outputs/ucrid_ablation_results_20260324.md)
- This paper-ready version: [ucrid_ablation_paper_ready_20260324.md](/mnt/data3/wzc/llm_oos_detection/outputs/ucrid_ablation_paper_ready_20260324.md)
