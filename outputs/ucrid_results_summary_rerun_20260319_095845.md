# UCRID CLINC150 实验结果汇总（rerun_20260319_095845）

## 主要指标对比

| 方法 | Accuracy | ID Accuracy | OOS Precision | OOS Recall | OOS F1 | LLM 调用率 |
|------|----------|-------------|---------------|------------|--------|-----------|
| Stage 1 (BERT only) | 86.35% | 96.07% | 95.73% | 42.60% | 0.5896 | - |
| Stage 1+2 (无 LLM) | **89.55%** | **95.96%** | **93.82%** | **60.70%** | **0.7371** | 8.1% |
| Stage 1+2+3 (LLaMA) | 88.85% | 95.84% | 92.73% | 57.40% | 0.7091 | 8.1% |

## 路由分布（测试集）

| 路由决策 | 比例 | 数量 |
|---------|------|------|
| Small model 直接输出 | 86.96% | 4783 |
| Direct OOS | 4.98% | 274 |
| LLM judge | 8.05% | 443 |

## 延迟统计

### Stage 1+2+3

| 指标 | 值 |
|------|----|
| P50 | 49.9 ms |
| P95 | 651.8 ms |
| P99 | 794.0 ms |

### Stage 1+2（无 LLM）

| 指标 | 值 |
|------|----|
| P50 | 0.8 ms |
| P95 | 0.8 ms |
| P99 | 1.0 ms |

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
| 约束满足 | 是 |

## 备注

- 本次完整实验使用 `wzcdev` conda 环境中的解释器：`/home/zcwang/anaconda3/envs/wzcdev/bin/python`。
- 新结果文件：`outputs/ucrid/clinc150_rerun_20260319_095845/ucrid_results.json`
- 新 Stage 1 结果：`outputs/stage1/ucrid_stage1_clinc150_rerun_20260319_095845/stage1_results.json`
- 旧文件 `outputs/ucrid_results_summary.md` 未覆盖。
