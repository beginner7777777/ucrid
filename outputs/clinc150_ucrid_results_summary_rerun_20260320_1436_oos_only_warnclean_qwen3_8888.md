# UCRID CLINC150 实验结果汇总（rerun_20260320_1436_oos_only_warnclean_qwen3_8888）

## 主要指标对比

| 方法 | Accuracy | ID Accuracy | OOS Precision | OOS Recall | OOS F1 | LLM 调用率 |
|------|----------|-------------|---------------|------------|--------|-----------|
| Stage 1 (BERT only) | 86.35% | 96.07% | 95.73% | 42.60% | 0.5896 | - |
| Stage 1+2 (无 LLM) | 89.55% | 95.96% | 93.82% | 60.70% | 0.7371 | 8.1% |
| Stage 1+2+3 (Qwen3-8B, `oos_only`) | 89.55% | 95.96% | 93.53% | 60.70% | 0.7362 | 8.1% |

## 路由分布（测试集）

| 路由决策 | 比例 | 数量 |
|---------|------|------|
| Small model 直接输出 | 86.96% | 4783 |
| Direct OOS | 4.98% | 274 |
| LLM judge | 8.05% | 443 |

## 延迟统计（Stage 1+2+3, `oos_only`）

| 指标 | 值 |
|------|----|
| P50 | 0.9 ms |
| P95 | 334.3 ms |
| P99 | 437.0 ms |

## 备注

- 运行环境：`/home/zcwang/anaconda3/envs/wzcdev/bin/python`。
- 配置文件：`configs/clinc150_config_openai_qwen3_8888.yaml`。
- 结果文件：`outputs/ucrid/clinc150_rerun_20260320_1436_oos_only_warnclean_qwen3_8888/ucrid_results.json`。
- 明细文件：`outputs/ucrid/clinc150_rerun_20260320_1436_oos_only_warnclean_qwen3_8888/ucrid_prediction_details.json`。
- 你指定的 `API_URL=http://127.0.0.1:8000/v1/completions` 当前不可用于该流程（`/v1/models` 返回 404，POST 也不支持），本次使用可用的 `http://127.0.0.1:8888/v1` 跑通 Qwen3-8B。
