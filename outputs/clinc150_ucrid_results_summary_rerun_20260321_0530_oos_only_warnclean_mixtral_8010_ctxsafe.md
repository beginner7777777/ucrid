# UCRID CLINC150 实验结果汇总（rerun_20260321_0530_oos_only_warnclean_mixtral_8010_ctxsafe）

## 主要指标对比

| 方法 | Accuracy | ID Accuracy | OOS Precision | OOS Recall | OOS F1 | LLM 调用率 |
|------|----------|-------------|---------------|------------|--------|-----------|
| Stage 1 (BERT only) | 86.35% | 96.07% | 95.73% | 42.60% | 0.5896 | - |
| Stage 1+2 (无 LLM) | 89.55% | 95.96% | 93.82% | 60.70% | 0.7371 | 8.1% |
| Stage 1+2+3 (Mixtral-8x7B, `oos_only`, ctxsafe) | **91.02%** | 95.91% | 93.62% | **69.00%** | **0.7945** | 8.1% |

## 延迟统计

| 指标 | Stage 1+2 | Stage 1+2+3 |
|------|-----------|-------------|
| P50 | 0.8 ms | 0.9 ms |
| P95 | 0.8 ms | 2593.5 ms |
| P99 | 0.9 ms | 3542.3 ms |

## Stage3 可用性诊断

| 指标 | 值 |
|------|----|
| LLM 路由样本数 | 443 |
| `llm_used` 数 | 179 |
| `llm_used_rate` | 40.4% |
| `llm_label=-1` 数 | 0 |
| `llm_label=OOS` 数 | 179 |

## 备注

- 配置文件：`configs/clinc150_config_openai_mixtral_8010_ctxsafe.yaml`。
- 本次适配点：为 Mixtral/vLLM 增加 `chat.completions -> completions` 自动回退，并缩短 Stage3 prompt 上下文以适应 512 上下文窗口。
- 结果文件：`outputs/ucrid/clinc150_rerun_20260321_0530_oos_only_warnclean_mixtral_8010_ctxsafe/ucrid_results.json`。
- 明细文件：`outputs/ucrid/clinc150_rerun_20260321_0530_oos_only_warnclean_mixtral_8010_ctxsafe/ucrid_prediction_details.json`。
