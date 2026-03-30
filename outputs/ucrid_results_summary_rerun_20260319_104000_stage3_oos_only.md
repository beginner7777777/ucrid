# UCRID CLINC150 实验结果汇总（rerun_20260319_104000_stage3_oos_only）

## 主要指标对比

| 方法 | Accuracy | ID Accuracy | OOS Precision | OOS Recall | OOS F1 | LLM 调用率 |
|------|----------|-------------|---------------|------------|--------|-----------|
| Stage 1 (BERT only) | 86.35% | 96.07% | 95.73% | 42.60% | 0.5896 | - |
| Stage 1+2 (无 LLM) | 89.55% | 95.96% | 93.82% | 60.70% | 0.7371 | 8.1% |
| Stage 1+2+3 (LLaMA, `all`) | 89.53% | 96.18% | 95.21% | 59.60% | 0.7331 | 8.1% |
| Stage 1+2+3 (LLaMA, `oos_only`) | **90.51%** | 95.96% | **94.29%** | **66.00%** | **0.7765** | 8.1% |

## 路由分布（测试集）

| 路由决策 | 比例 | 数量 |
|---------|------|------|
| Small model 直接输出 | 86.96% | 4783 |
| Direct OOS | 4.98% | 274 |
| LLM judge | 8.05% | 443 |

## 延迟统计（Stage 1+2+3, `oos_only`）

| 指标 | 值 |
|------|----|
| P50 | 16.1 ms |
| P95 | 211.2 ms |
| P99 | 253.7 ms |

## 最优路由阈值（验证集搜索）

| 参数 | 值 |
|------|----|
| alpha | 0.7 |
| temperature | 0.8129 |
| tau_accept | 0.4 |
| tau_reject | 0.7 |
| delta | 1.0 |
| Val OOS F1 | 0.7021 |
| Val Accuracy | 95.19% |
| Val LLM 调用率 | 3.2% |
| LLM 接纳策略 | `oos_only` |

## 备注

- 本次完整实验使用 `wzcdev` conda 环境中的解释器：`/home/zcwang/anaconda3/envs/wzcdev/bin/python`。
- 新结果文件：`outputs/ucrid/clinc150_rerun_20260319_104000_stage3_oos_only/ucrid_results.json`
- 新明细文件：`outputs/ucrid/clinc150_rerun_20260319_104000_stage3_oos_only/ucrid_prediction_details.json`
- 旧文件 `outputs/ucrid_results_summary.md` 未覆盖。
