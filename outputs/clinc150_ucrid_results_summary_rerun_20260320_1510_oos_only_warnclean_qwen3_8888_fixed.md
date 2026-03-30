# UCRID CLINC150 实验结果汇总（rerun_20260320_1510_oos_only_warnclean_qwen3_8888_fixed）

## 主要指标对比

| 方法 | Accuracy | ID Accuracy | OOS Precision | OOS Recall | OOS F1 | LLM 调用率 |
|------|----------|-------------|---------------|------------|--------|-----------|
| Stage 1 (BERT only) | 86.35% | 96.07% | 95.73% | 42.60% | 0.5896 | - |
| Stage 1+2 (无 LLM) | 89.55% | 95.96% | 93.82% | 60.70% | 0.7371 | 8.1% |
| Stage 1+2+3 (Qwen3-8B fixed, `oos_only`) | **92.40%** | 95.91% | 93.76% | **76.60%** | **0.8431** | 8.1% |

## 延迟统计

| 指标 | Stage 1+2 | Stage 1+2+3 |
|------|-----------|-------------|
| P50 | 0.8 ms | 0.9 ms |
| P95 | 0.8 ms | 117.0 ms |
| P99 | 0.9 ms | 157.4 ms |

## Stage3 可用性诊断

| 指标 | 值 |
|------|----|
| LLM 路由样本数 | 443 |
| `llm_used` 数 | 305 |
| `llm_used_rate` | 68.9% |
| `llm_label=-1` 数 | 0 |
| `llm_label=OOS` 数 | 305 |

## 备注

- 配置文件：`configs/clinc150_config_openai_qwen3_8888_fixed.yaml`。
- 本次修复点：关闭 Qwen3 thinking 输出 + 增强解析 + 提高 `max_tokens`。
- 结果文件：`outputs/ucrid/clinc150_rerun_20260320_1510_oos_only_warnclean_qwen3_8888_fixed/ucrid_results.json`。
- 明细文件：`outputs/ucrid/clinc150_rerun_20260320_1510_oos_only_warnclean_qwen3_8888_fixed/ucrid_prediction_details.json`。
