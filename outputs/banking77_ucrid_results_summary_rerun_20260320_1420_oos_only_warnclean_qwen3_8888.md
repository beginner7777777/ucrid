# UCRID Banking77 实验结果汇总（rerun_20260320_1420_oos_only_warnclean_qwen3_8888）

## 主要指标对比

| 方法 | Accuracy | ID Accuracy | OOS Precision | OOS Recall | OOS F1 | LLM 调用率 |
|------|----------|-------------|---------------|------------|--------|-----------|
| Stage 1 (BERT only) | 79.78% | 92.73% | 0.00% | 0.00% | 0.0000 | - |
| Stage 1+2 (无 LLM) | 89.94% | 92.47% | 95.38% | 74.40% | 0.8360 | 7.2% |
| Stage 1+2+3 (Qwen3-8B, `oos_only`) | 90.00% | 92.47% | 95.41% | 74.80% | 0.8386 | 7.2% |

## 延迟统计

| 指标 | Stage 1+2 | Stage 1+2+3 |
|------|-----------|-------------|
| P50 | 0.7 ms | 59.4 ms |
| P95 | 0.7 ms | 97.4 ms |
| P99 | 2.3 ms | 100.6 ms |

## 备注

- 运行环境：`/home/zcwang/anaconda3/envs/wzcdev/bin/python`。
- LLM 服务：`OPENAI_BASE_URL=http://127.0.0.1:8888/v1`，model=`Qwen/Qwen3-8B`。
- 配置文件：`configs/banking77_config_openai_qwen3_8888.yaml`。
- 结果文件：`outputs/ucrid/banking77_rerun_20260320_1420_oos_only_warnclean_qwen3_8888/ucrid_results.json`。
- 明细文件：`outputs/ucrid/banking77_rerun_20260320_1420_oos_only_warnclean_qwen3_8888/ucrid_prediction_details.json`。
- 你指定的 `http://127.0.0.1:8000/v1/completions` 当前返回 `501 Unsupported method ('POST')`，因此本次改用可用端口 `8888`。
