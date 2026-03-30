# UCRID CLINC150 实验结果汇总（rerun_20260320_114500_warnclean_retry）

## 主要指标

| 方法 | Accuracy | ID Accuracy | OOS Precision | OOS Recall | OOS F1 | LLM 调用率 |
|------|----------|-------------|---------------|------------|--------|-----------|
| Stage 1 (BERT only) | 86.35% | 96.07% | 95.73% | 42.60% | 0.5896 | - |
| Stage 1+2 (无 LLM) | 89.55% | 95.96% | 93.82% | 60.70% | 0.7371 | 8.1% |
| Stage 1+2+3 (LLaMA, `oos_only`) | **90.51%** | 95.96% | **94.29%** | **66.00%** | **0.7765** | 8.1% |

## 路由分布（测试集）

| 路由决策 | 比例 | 数量 |
|---------|------|------|
| Small model 直接输出 | 86.96% | 4783 |
| Direct OOS | 4.98% | 274 |
| LLM judge | 8.05% | 443 |

## 延迟统计（Stage 1+2+3）

| 指标 | 值 |
|------|----|
| P50 | 0.9 ms |
| P95 | 222.2 ms |
| P99 | 293.6 ms |

## 备注

- 本次运行使用临时配置 `configs/clinc150_config_warnclean_tmp.yaml`（`num_workers=0`, `inference.batch_size=16`）以规避 CUDA OOM。
- 新结果文件：`outputs/ucrid/clinc150_rerun_20260320_114500_warnclean_retry/ucrid_results.json`
- 新明细文件：`outputs/ucrid/clinc150_rerun_20260320_114500_warnclean_retry/ucrid_prediction_details.json`
- 原始 summary 文件均未覆盖。
