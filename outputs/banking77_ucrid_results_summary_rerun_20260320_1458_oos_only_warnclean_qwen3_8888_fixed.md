# UCRID Banking77 实验结果汇总（rerun_20260320_1458_oos_only_warnclean_qwen3_8888_fixed）

## 主要指标对比

| 方法 | Accuracy | ID Accuracy | OOS Precision | OOS Recall | OOS F1 | LLM 调用率 |
|------|----------|-------------|---------------|------------|--------|-----------|
| Stage 1 (BERT only) | 79.78% | 92.73% | 0.00% | 0.00% | 0.0000 | - |
| Stage 1+2 (无 LLM) | 89.94% | 92.47% | 95.38% | 74.40% | 0.8360 | 7.2% |
| Stage 1+2+3 (Qwen3-8B fixed, `oos_only`) | **92.65%** | 92.44% | 94.95% | **94.00%** | **0.9447** | 7.2% |

## 延迟统计

| 指标 | Stage 1+2 | Stage 1+2+3 |
|------|-----------|-------------|
| P50 | 0.7 ms | 24.0 ms |
| P95 | 0.7 ms | 39.2 ms |
| P99 | 2.3 ms | 44.0 ms |

## Stage3 可用性诊断

| 指标 | 值 |
|------|----|
| LLM 路由样本数 | 258 |
| `llm_used` 数 | 105 |
| `llm_used_rate` | 40.7% |
| `llm_label=-1` 数 | 0 |
| `llm_label=OOS` 数 | 105 |

## 备注

- 配置文件：`configs/banking77_config_openai_qwen3_8888_fixed.yaml`。
- 本次修复点：关闭 Qwen3 thinking 输出 + 增强解析 + 提高 `max_tokens`。
- 结果文件：`outputs/ucrid/banking77_rerun_20260320_1458_oos_only_warnclean_qwen3_8888_fixed/ucrid_results.json`。
- 明细文件：`outputs/ucrid/banking77_rerun_20260320_1458_oos_only_warnclean_qwen3_8888_fixed/ucrid_prediction_details.json`。
