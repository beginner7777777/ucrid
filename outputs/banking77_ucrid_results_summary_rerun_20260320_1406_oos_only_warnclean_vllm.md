# UCRID Banking77 实验结果汇总（rerun_20260320_1406_oos_only_warnclean_vllm）

## 主要指标对比

| 方法 | Accuracy | ID Accuracy | OOS Precision | OOS Recall | OOS F1 | LLM 调用率 |
|------|----------|-------------|---------------|------------|--------|-----------|
| Stage 1 (BERT only) | 79.78% | 92.73% | 0.00% | 0.00% | 0.0000 | - |
| Stage 1+2 (无 LLM) | 89.94% | 92.47% | 95.38% | 74.40% | 0.8360 | 7.2% |
| Stage 1+2+3 (LLM, `oos_only`) | **92.60%** | 92.37% | 94.00% | **94.00%** | **0.9400** | 7.2% |

## 路由分布（测试集）

| 路由决策 | 比例 | 数量 |
|---------|------|------|
| Small model 直接输出 | 81.90% | 2932 |
| Direct OOS | 10.89% | 390 |
| LLM judge | 7.21% | 258 |

## 延迟统计

| 指标 | Stage 1+2 | Stage 1+2+3 (`oos_only`) |
|------|-----------|---------------------------|
| P50 | 0.7 ms | 23.0 ms |
| P95 | 0.7 ms | 37.7 ms |
| P99 | 3.4 ms | 41.6 ms |

## 最优路由阈值（验证集搜索）

| 参数 | 值 |
|------|----|
| alpha | 0.7 |
| temperature | 0.8755 |
| tau_accept | 0.4 |
| tau_reject | 0.7 |
| delta | 1.0 |
| Val OOS F1 | 0.7978 |
| Val Accuracy | 91.74% |
| Val LLM 调用率 | 6.0% |
| LLM 接纳策略 | `oos_only` |

## 备注

- 本次完整实验使用 `wzcdev` conda 环境解释器：`/home/zcwang/anaconda3/envs/wzcdev/bin/python`。
- Stage 3 使用 OpenAI 兼容本地 vLLM 服务（`OPENAI_BASE_URL=http://127.0.0.1:8003/v1`, model=`rule_v4`）。
- 配置文件：`configs/banking77_config_openai_vllm.yaml`。
- 结果文件：`outputs/ucrid/banking77_rerun_20260320_1406_oos_only_warnclean_vllm/ucrid_results.json`。
- 明细文件：`outputs/ucrid/banking77_rerun_20260320_1406_oos_only_warnclean_vllm/ucrid_prediction_details.json`。
- 旧结果文件未覆盖。
