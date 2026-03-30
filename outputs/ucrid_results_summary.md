# UCRID CLINC150 实验结果汇总

## 主要指标对比

| 方法 | Accuracy | ID Accuracy | OOS Precision | OOS Recall | OOS F1 | LLM 调用率 |
|------|----------|-------------|---------------|------------|--------|-----------|
| Stage 1 (BERT only) | 86.35% | 96.07% | 95.73% | 42.60% | 0.5896 | - |
| Stage 1+2 (无 LLM) | 84.16% | 93.40% | 89.31% | 42.60% | 0.5768 | 11.4% |
| Stage 1+2+3 (LLaMA) | **86.60%** | **95.67%** | **89.63%** | **45.80%** | **0.6062** | 11.4% |

## 路由分布（Stage 2）

| 路由决策 | 比例 |
|---------|------|
| Small model 直接输出 | 79.9% |
| Direct OOS | 8.7% |
| LLM judge | 11.4% |

## 延迟统计（Stage 1+2+3）

| 指标 | 值 |
|------|----|
| P50 | 50.4 ms |
| P95 | 787.4 ms |
| P99 | 1030.9 ms |

## 最优路由阈值（验证集搜索）

| 参数 | 值 |
|------|----|
| τ_accept | 0.4 |
| τ_reject | 0.7 |
| Val OOS F1 | 0.7177 |
| Val LLM 调用率 | 5.2% |

## 训练配置

| 参数 | 值 |
|------|----|
| 模型 | BERT-base-uncased |
| LLM Judge | LLaMA-3-8B-Instruct (4bit) |
| 训练轮数 | 10 epochs |
| 批大小 | 64 |
| 学习率 | 2e-5 |
| λ_contrastive | 0.3 |
| λ_boundary | 0.1 |
| 训练策略 | CE-only (1-3) → CE+SupCon (4-6) → CE+SupCon+Boundary (7-10) |
| 数据集 | CLINC150 (train=15200, val=3100, test=5500) |
