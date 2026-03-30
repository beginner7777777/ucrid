# UCRID 消融实验结果（2026-03-24）

## 1. 实验目的

本次消融实验针对当前主流程 `run_ucrid.py` 做结构化拆解，重点回答三个问题：

1. Stage 2 的双信号路由是否真的有效？
2. 熵信号和原型距离信号哪个贡献更大？
3. Stage 3 的 LLM 仲裁相对 Stage 1+2 是否还有额外收益？

为了保证口径统一，本次消融全部基于当前最佳主流程：

- CLINC150 使用 `Qwen3-8B fixed + oos_only`
- Banking77 使用 `Qwen3-8B fixed + oos_only`

其中：

- `Stage1 only` 和 `Stage1+2 (full router)` 直接复用已有最佳实验中的同口径结果。
- 新补跑了 4 个无 LLM 的路由消融：
  - `entropy-only router`
  - `distance-only router`
  - CLINC150 一组
  - Banking77 一组

---

## 2. 消融设置

### 2.1 CLINC150

- `Stage1 only`：仅小模型分类
- `Stage1+2 (full router)`：完整双阈值路由，但不启用 LLM
- `w/o distance signal (entropy-only)`：固定 `alpha=1.0`
- `w/o entropy signal (distance-only)`：固定 `alpha=0.0`
- `Full UCRID`：Stage1 + Stage2 + Stage3（Qwen3-8B fixed, `oos_only`）

### 2.2 Banking77

- `Stage1 only`：仅小模型分类
- `Stage1+2 (full router)`：完整双阈值路由，但不启用 LLM
- `w/o distance signal (entropy-only)`：固定 `alpha=1.0`
- `w/o entropy signal (distance-only)`：固定 `alpha=0.0`
- `Full UCRID`：Stage1 + Stage2 + Stage3（Qwen3-8B fixed, `oos_only`）

---

## 3. CLINC150 消融结果

| 设置 | Accuracy | ID Accuracy | OOS Precision | OOS Recall | OOS F1 | LLM Call Rate |
|------|----------|-------------|---------------|------------|--------|---------------|
| Stage1 only | 86.35% | 96.07% | 95.73% | 42.60% | 0.5896 | - |
| Stage1+2 (full router) | 89.55% | 95.96% | 93.82% | 60.70% | 0.7371 | 8.1% |
| w/o distance signal (entropy-only) | 89.11% | 95.98% | 94.33% | 58.20% | 0.7199 | 5.9% |
| w/o entropy signal (distance-only) | 91.07% | 95.76% | 91.50% | 70.00% | 0.7932 | 9.6% |
| Full UCRID | **92.40%** | 95.91% | 93.76% | **76.60%** | **0.8431** | 8.1% |

### 3.1 结果分析

- `Stage1 only -> Stage1+2`：
  OOS F1 从 `0.5896` 提升到 `0.7371`，说明仅靠双阈值路由和直接 OOS 判定，已经能显著增强 OOS 检测能力。

- `entropy-only`：
  OOS F1 为 `0.7199`，低于完整 router 的 `0.7371`。这说明只看 softmax 熵不够，模型的分类不确定性无法完全覆盖 OOS 结构信息。

- `distance-only`：
  OOS F1 为 `0.7932`，明显高于 `entropy-only`，也高于 `Stage1+2(full router)`。
  这说明在 CLINC150 上，**原型距离信号比熵信号更重要**，因为该数据集上的 OOS 更容易表现为“远离所有已知意图簇”。

- `full router` vs `distance-only`：
  完整 router 的 Stage2 OOS F1 反而低于 `distance-only`，说明当前的 `alpha=0.7` 搜索结果更偏向在性能约束下控制 LLM 调用率，而不是单独最大化 Stage2 OOS F1。

- `Full UCRID`：
  最终达到 `0.8431`，说明 Stage3 LLM 仲裁在 CLINC150 上仍然提供了显著额外收益。

### 3.2 CLINC150 结论

- CLINC150 上最关键的 Stage2 信号是 **prototype distance**。
- 熵信号单独使用时效果较弱，但作为融合项有助于平衡误判和调用率。
- Stage3 在 CLINC150 上收益最大，说明该数据集更依赖 LLM 对不确定区域的精判能力。

---

## 4. Banking77 消融结果

| 设置 | Accuracy | ID Accuracy | OOS Precision | OOS Recall | OOS F1 | LLM Call Rate |
|------|----------|-------------|---------------|------------|--------|---------------|
| Stage1 only | 79.78% | 92.73% | 0.00% | 0.00% | 0.0000 | - |
| Stage1+2 (full router) | 89.94% | **92.47%** | 95.38% | 74.40% | 0.8360 | 7.2% |
| w/o distance signal (entropy-only) | 89.61% | 92.53% | **95.72%** | 71.60% | 0.8192 | 6.2% |
| w/o entropy signal (distance-only) | 91.17% | 92.34% | 91.50% | 84.00% | 0.8759 | 11.9% |
| Full UCRID | **92.65%** | 92.44% | 94.95% | **94.00%** | **0.9447** | 7.2% |

### 4.1 结果分析

- `Stage1 only`：
  OOS F1 为 `0`，说明小模型本身几乎不会主动发现 Banking77 中的 OOS。

- `Stage1 only -> Stage1+2`：
  OOS F1 从 `0` 直接跃升到 `0.8360`，说明路由器和直接 OOS 判定对 Banking77 非常关键。

- `entropy-only`：
  OOS F1 为 `0.8192`，接近完整 router，但仍有下降。说明熵信号在 Banking77 上是有效的，但不够充分。

- `distance-only`：
  OOS F1 为 `0.8759`，高于完整 Stage2，也高于 `entropy-only`。这表明 Banking77 上同样是 **prototype distance** 更强。

- `distance-only` 的代价：
  它的 `LLM call rate` 达到 `11.9%`，明显高于完整 router 的 `7.2%`。这说明距离信号虽然更激进，但会把更多样本送入不确定区域。

- `Full UCRID`：
  最终达到 `0.9447`，说明 Stage3 对 Banking77 同样有效，而且增益比 CLINC150 更大。

### 4.2 Banking77 结论

- Banking77 上，Stage2 的核心仍然是 **distance signal**。
- 但与 CLINC150 相比，Banking77 的完整 router 更明显地在“性能”和“调用率”之间做了折中。
- Stage3 对 Banking77 的收益非常显著，尤其把 OOS recall 从 `74.4%` 拉到 `94.0%`。

---

## 5. 总体结论

从两个数据集的消融结果看，可以得到 4 个稳定结论：

1. **Stage2 是必要的**
   无论 CLINC150 还是 Banking77，仅从 Stage1 到 Stage1+2 都带来了大幅提升。

2. **prototype distance 是更强的路由信号**
   在两个数据集上，`distance-only` 都明显优于 `entropy-only`。

3. **完整 router 的价值不是只追求 Stage2 最优**
   当前双信号融合和验证集搜索，实际上在优化“性能约束下的 LLM 调用率”，而不只是局部最大化 Stage2 OOS F1。

4. **Stage3 仍然是最终性能上限的关键**
   两个数据集上，最终最优结果都来自 `Full UCRID`，说明 LLM 仲裁对不确定区域仍有不可替代的贡献。

---

## 6. 论文可写的结论表达

可以直接写成下面这个结论：

> Ablation results on both CLINC150 and Banking77 show that the dual-threshold router is essential for enabling practical OOS detection, while prototype distance contributes more than entropy as the dominant routing signal. Nevertheless, the best overall performance is achieved only when the router is further combined with Stage-3 LLM arbitration, indicating that uncertainty-aware routing and candidate-constrained LLM judgment are complementary rather than redundant.

---

## 7. 本次新增实验结果文件

### CLINC150

- entropy-only：
  [ucrid_results.json](/mnt/data3/wzc/llm_oos_detection/outputs/ucrid/clinc150_ablation_20260324_entropy_only/ucrid_results.json)
- distance-only：
  [ucrid_results.json](/mnt/data3/wzc/llm_oos_detection/outputs/ucrid/clinc150_ablation_20260324_distance_only/ucrid_results.json)

### Banking77

- entropy-only：
  [ucrid_results.json](/mnt/data3/wzc/llm_oos_detection/outputs/ucrid/banking77_ablation_20260324_entropy_only/ucrid_results.json)
- distance-only：
  [ucrid_results.json](/mnt/data3/wzc/llm_oos_detection/outputs/ucrid/banking77_ablation_20260324_distance_only/ucrid_results.json)
