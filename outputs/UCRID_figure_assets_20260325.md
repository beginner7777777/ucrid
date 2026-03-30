# UCRID Figure Assets for Paper

## Recommended Figure Mapping

### Figure 1: Overall pipeline

- Image: [ucrid_pipeline_diagram_20260321.png](/mnt/data3/wzc/llm_oos_detection/outputs/figures/ucrid_pipeline_diagram_20260321.png)
- Suggested caption:
  `Overview of UCRID. A lightweight encoder first processes every query. The dual-signal router then separates confident in-domain samples, obvious OOS samples, and ambiguous samples that are escalated to Stage-3 LLM arbitration.`

### Figure 2: Training vs inference

- Image: [ucrid_dual_column_train_infer_20260321.png](/mnt/data3/wzc/llm_oos_detection/outputs/figures/ucrid_dual_column_train_infer_20260321.png)
- Suggested caption:
  `Two-view illustration of UCRID. Left: training phase with cross-entropy, supervised contrastive loss, and boundary-aware loss. Right: inference phase with uncertainty-aware routing and candidate-constrained LLM judgment.`

### Figure 3: Algorithmic flowchart

- Image: [ucrid_algorithm_flowchart_20260321.png](/mnt/data3/wzc/llm_oos_detection/outputs/figures/ucrid_algorithm_flowchart_20260321.png)
- Suggested caption:
  `Algorithmic flow of UCRID inference. The router computes entropy and prototype distance, applies dual-threshold decisions, and forwards only ambiguous cases to the LLM.`

### Figure 4: Ablation chart

- Image: [ablation_chart.png](/mnt/data3/wzc/llm_oos_detection/outputs/figures/ablation_chart.png)
- Suggested caption:
  `Ablation comparison on CLINC150 and Banking77. Prototype distance contributes more than entropy, while the full UCRID cascade achieves the best final OOS F1.`

### Optional Figure 5: t-SNE visualization

- Image: [tsne_visualization.png](/mnt/data3/wzc/llm_oos_detection/outputs/figures/tsne_visualization.png)
- Suggested caption:
  `Embedding-space visualization illustrating the separation between in-domain clusters and OOS samples after prototype- and boundary-aware representation learning.`

## Suggested Paper Placement

- `Figure 1` in Method overview
- `Figure 2` at the beginning of Method or Appendix
- `Figure 3` near the router / inference subsection
- `Figure 4` in Ablation Study
- `Figure 5` in Discussion or Appendix

## File Notes

- All figures already exist under [outputs/figures](/mnt/data3/wzc/llm_oos_detection/outputs/figures).
- If you compile the LaTeX from the same `outputs` directory, use relative paths such as `figures/ucrid_pipeline_diagram_20260321.png`.
