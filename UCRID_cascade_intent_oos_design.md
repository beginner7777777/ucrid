# UCRID: Uncertainty-aware Cascade Routing for Intent Detection and OOS Detection

**版本**: v1.0
**创建时间**: 2026-03-18
**状态**: 设计阶段

---

## 1. 研究动机

| 方法 | 优势 | 劣势 |
|------|------|------|
| 纯小模型 (BERT-base) | 快速 (~15ms), 成本低 | OOS 检测能力弱, 准确率有限 |
| 纯大模型 (GPT-4) | 准确率高, OOS 能力强 | 慢 (~2000ms), API 成本高 |
| 级联协同 | 兼顾效率与准确率 | 需要精心设计路由策略 |

**核心洞察**: 大多数查询小模型可以高置信度处理；只有不确定区域才需要 LLM。
通过对比学习优化嵌入空间 + 边界损失显式建模 OOS，可以让小模型本身具备 OOS 感知能力，
从而减少 LLM 调用率。

---

## 2. 整体架构

```
用户输入 u
    ↓
[Stage 1] BERT-base 对比学习分类器
    → 输出: 意图概率分布 P(y|u), 嵌入向量 h_u
    ↓
[Stage 2] 双阈值自适应路由
    → 计算不确定性分数 s(u)
    ↓
 s(u) ≤ τ_accept ?
  /       \
是          否
↓           ↓
直接输出    s(u) ≥ τ_reject 且 d_min > Δ ?
Top-1意图    /       \
(INS)      是          否
            ↓           ↓
         标记OOS    [Stage 3] LLM精判
         (高不确定    GPT-4/Claude few-shot
          +远离所有     ↓
          原型)     输出: 意图 或 OOS
```

---

## 3. Stage 1: 对比学习增强的小模型

### 3.1 模型选择

- 主选: BERT-base (110M 参数)
- 轻量备选: DistilBERT (66M 参数, BERT-base 的 40%)

### 3.2 训练损失函数

```
L = L_CE + λ_s · L_SupCon + λ_b · L_Boundary
```

**L_CE: 标准交叉熵**
```
L_CE = -Σ_i y_i · log P(y_i | u_i; θ)
```
保证基本分类准确率。

**L_SupCon: 监督对比损失**
```
L_SupCon = Σ_i (-1/|P(i)|) Σ_{p∈P(i)} log [
    exp(sim(h_i, h_p) / τ_c)
    / Σ_{a∈A(i)} exp(sim(h_i, h_a) / τ_c)
]
```
- P(i): 与样本 i 同类的正样本集
- A(i): batch 中所有非 i 样本集
- τ_c: 对比学习温度参数 (推荐 0.07)
- sim: 余弦相似度

作用: 拉近同类样本嵌入，推远异类样本嵌入，形成紧凑的意图簇。

**L_Boundary: OOS 边界损失**
```
L_Boundary = (1/|B_OOS|) Σ_{x∈B_OOS} max(0, Δ - d(h_x, p_{c*}))
```
- B_OOS: 训练集中的 OOS 样本集合
- p_{c*} = argmin_c d(h_x, p_c): 离 OOS 样本最近的意图原型
- p_c: 意图 c 的原型向量 (该类训练样本嵌入的均值)
- Δ: 边界间距 (推荐 1.0)
- d: 欧氏距离

作用: 强制 OOS 样本的嵌入远离所有意图原型，在嵌入空间中形成清晰的 OOS 区域。

### 3.3 推荐超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| λ_s | 0.3 | SupCon 损失权重 |
| λ_b | 0.1 | Boundary 损失权重 |
| τ_c | 0.07 | 对比学习温度 |
| Δ | 1.0 | OOS 边界间距 |
| 学习率 | 2e-5 | AdamW |
| Batch size | 32 | |
| 训练轮数 | 10 | |

> 注: λ_s, λ_b 需在验证集上调优，建议搜索范围 λ_s ∈ {0.1, 0.3, 0.5}, λ_b ∈ {0.05, 0.1, 0.2}

### 3.4 意图原型更新策略

```python
# 每个 epoch 结束后更新原型
for intent_c in all_intents:
    embeddings_c = [h_x for x in train_set if label(x) == intent_c]
    prototype[intent_c] = mean(embeddings_c)
```

---

## 4. Stage 2: 双阈值自适应路由

### 4.1 不确定性分数计算

融合两种互补信号:

```python
# 信号1: 校准后的 Softmax 熵 (捕捉分类不确定性)
H(u) = -Σ_y P(y|u) · log P(y|u)

# 信号2: 到最近意图原型的距离 (捕捉 OOS 可能性)
d_min(u) = min_c ||h_u - p_c||_2

# 归一化到 [0, 1]
H_norm(u) = (H(u) - H_min) / (H_max - H_min)
d_norm(u) = (d_min(u) - d_min_global) / (d_max_global - d_min_global)

# 融合不确定性分数
s(u) = α · H_norm(u) + (1 - α) · d_norm(u)
```

推荐 α = 0.5，可在验证集上搜索 α ∈ {0.3, 0.5, 0.7}。

### 4.2 双阈值路由逻辑

```python
def route(u, s_u, d_min_u, threshold_accept, threshold_reject, delta):
    if s_u <= threshold_accept:
        # 高置信度: 直接接受小模型预测
        return "small_model", top1_intent(u)

    elif s_u >= threshold_reject and d_min_u > delta:
        # 高不确定性 + 远离所有原型: 直接判定 OOS
        return "direct_oos", "OOS"

    else:
        # 不确定区域: 路由到 LLM 精判
        return "llm", top_k_intents(u, k=3)
```

| 条件 | 决策 | 含义 |
|------|------|------|
| s(u) ≤ τ_accept | 直接输出小模型 Top-1 | 高置信度 INS |
| s(u) ≥ τ_reject 且 d_min > Δ | 直接标记 OOS | 明确 OOS，无需 LLM |
| 其他 | 路由到 LLM | 不确定区域 |

### 4.3 阈值搜索策略

在验证集上进行网格搜索:
- τ_accept ∈ {0.2, 0.3, 0.4}
- τ_reject ∈ {0.7, 0.8, 0.9}
- 优化目标: 在 OOS F1 下降 ≤ 1% 的约束下，最小化 LLM 调用率

---

## 5. Stage 3: LLM 精判

### 5.1 Prompt 模板

```
You are an intent classifier for a task-oriented dialogue system.

Given a user query and a list of candidate intents with descriptions,
determine the most likely intent. If the query does not match any
candidate intent, respond with "OOS".

## Candidate Intents:
{top_k_intents_with_definitions}

## Few-shot Examples:
{3-5 examples per candidate intent}
{2-3 OOS examples}

## User Query:
"{query}"

## Your Response (intent name or "OOS"):
```

### 5.2 关键设计

- **Top-k 候选**: k=3，来自小模型输出的 Top-3 意图（参考 UDRIL）
- **意图定义**: 每个意图附带自然语言描述，可用 GPT-3.5 自动生成
  - 输入: 意图名称 + 5 个训练样本
  - 要求: 生成一个区分性定义，突出与相似意图的差异
- **Few-shot 示例**: 动态选择，从训练集中选与当前查询最相似的样本
- **候选顺序**: 每次随机打乱，防止位置偏见（参考 UDRIL）

### 5.3 LLM 选择建议

| LLM | 优势 | 适用场景 |
|-----|------|---------|
| GPT-4 | 准确率最高 | 准确率优先 |
| Claude-3.5-Sonnet | 速度快，成本低 | 平衡场景 |
| Llama-3.1-8B (LoRA) | 可本地部署，无 API 成本 | 成本敏感 / 数据隐私 |

---

## 6. 实验设计

### 6.1 数据集

| 数据集 | INS 意图数 | 规模 | OOS 比例 | 用途 |
|--------|-----------|------|---------|------|
| CLINC150 | 150 | 23,700 | 25% | 主实验（多领域） |
| Banking77 | 77 | 13,083 | 20% | 主实验（细粒度） |
| HINT3-Curekart | 21-27 | ~1000 | ~40% | 真实 OOS 场景 |
| HINT3-SOFMattress | 21-25 | ~1000 | ~40% | 真实 OOS 场景 |
| HINT3-PowerPlay11 | 57 | ~1000 | ~68% | 高 OOS 比例场景 |

### 6.2 对比方法

| 方法 | 类型 | 说明 |
|------|------|------|
| BERT-base + MSP | 纯小模型 | Softmax 最大概率阈值 |
| BERT-base + ADB | 纯小模型 | 自适应决策边界 |
| GPT-4 Zero-shot | 纯大模型 | 全量 LLM 推理 |
| GPT-4 Few-shot | 纯大模型 | 全量 LLM + few-shot |
| Hybrid (τ=0.8) | 级联 | 简单置信度路由 (Ghriss et al., 2024) |
| UDRIL-FT (high) | 级联 | NNK-Means + LoRA 微调 (ACL 2025) |
| OOS-Collab | 训练协同 | LLM 数据增强 + 蒸馏 (Feng et al., 2024) |
| **UCRID (Ours)** | **级联** | **对比学习 + 双阈值路由** |

### 6.3 评估指标

**性能指标**:
- OOS F1 (主要指标)
- OOS Precision, OOS Recall
- INS Accuracy
- Overall Accuracy

**效率指标**:
- LLM 调用率 (%)
- API 成本估算 ($/1000 queries)
- 延迟分析: P50, P95, P99 (ms)

### 6.4 成本-性能权衡分析

```
设:
  N = 总查询数
  r = LLM 路由率
  C_s = 小模型单次推理成本 (~0.0001$)
  C_l = LLM 单次 API 成本 (~0.01$ for GPT-4)

总成本 = N · C_s + N · r · C_l
成本节省率 = 1 - (C_s + r · C_l) / C_l = 1 - r - C_s/C_l ≈ 1 - r

加速比 = T_LLM_only / (T_small + r · T_LLM)
       ≈ 2000 / (15 + r · 2000)
```

**预期结果**:

| 路由策略 | LLM 调用率 | 预期 OOS F1 | 成本节省 | P50 延迟 | P95 延迟 |
|---------|-----------|------------|---------|---------|---------|
| 全量 LLM | 100% | ~89% | 0% | ~2000ms | ~3000ms |
| 高路由 (τ_a=0.2) | ~40% | ~91% | ~60% | ~25ms | ~2100ms |
| 中路由 (τ_a=0.3) | ~25% | ~90% | ~75% | ~20ms | ~2050ms |
| 低路由 (τ_a=0.4) | ~15% | ~88% | ~85% | ~18ms | ~2030ms |
| 纯小模型 | 0% | ~86% | ~100% | ~15ms | ~20ms |

### 6.5 消融实验

| 实验组 | 目的 |
|--------|------|
| w/o L_SupCon (λ_s=0) | 验证对比学习的贡献 |
| w/o L_Boundary (λ_b=0) | 验证边界损失的贡献 |
| 单阈值 vs 双阈值 | 验证双阈值路由的优势 |
| 仅熵 vs 仅距离 vs 融合 | 验证不确定性信号的选择 |
| Top-1 vs Top-3 vs Top-5 候选 | LLM 输入候选数的影响 |
| λ_s ∈ {0.1, 0.3, 0.5} | 损失权重敏感性 |
| λ_b ∈ {0.05, 0.1, 0.2} | 损失权重敏感性 |
| α ∈ {0.3, 0.5, 0.7} | 不确定性融合权重敏感性 |

### 6.6 延迟分析方法

```python
import numpy as np

def analyze_latency(latencies):
    return {
        "P50": np.percentile(latencies, 50),
        "P95": np.percentile(latencies, 95),
        "P99": np.percentile(latencies, 99),
        "mean": np.mean(latencies),
        "std": np.std(latencies)
    }

# 分别统计三类请求的延迟
latency_small = [...]   # 小模型直接处理
latency_oos_direct = [...]  # 直接判定 OOS
latency_llm = [...]     # 路由到 LLM

# 整体延迟 = 加权平均
overall_latency = (
    (1 - r_llm - r_oos_direct) * latency_small +
    r_oos_direct * latency_oos_direct +
    r_llm * latency_llm
)
```

---

## 7. 与现有工作的对比

| 维度 | UDRIL (ACL 2025) | Hybrid (2024) | OOS-Collab (2024) | UCRID (Ours) |
|------|-----------------|---------------|-------------------|--------------|
| 小模型训练 | Focal Loss | 标准 CE | CE + KL + 对比 | CE + SupCon + Boundary |
| OOS 建模 | 隐式 (LLM 判断) | 无 | LLM 生成 OOS 数据 | 显式边界损失 |
| 不确定性 | NNK-Means 重构误差 | Softmax max | 无路由 | 熵 + 原型距离融合 |
| 路由策略 | 单阈值 | 单阈值 | 无路由 | 双阈值 + OOS 直判 |
| LLM 使用 | LoRA 微调 8B | API few-shot | 训练时生成数据 | API few-shot |
| 推理时 LLM | 部分样本 | 部分样本 | 无 | 部分样本 |
| 训练成本 | 低 | 低 | 高 (大量 LLM 调用) | 低 |

**核心差异化**:
1. L_Boundary 在训练阶段显式建模 OOS 边界，小模型本身具备 OOS 感知能力
2. 双阈值设计允许对"明确 OOS"直接判定，减少不必要的 LLM 调用
3. 熵 + 原型距离的融合不确定性信号，比单一 Softmax 置信度更鲁棒

---

## 8. 实现路线图

### Phase 1: 小模型训练 (Week 1-2)
- [ ] 数据预处理 (CLINC150, Banking77, HINT3)
- [ ] 实现 L_SupCon + L_Boundary 损失函数
- [ ] 训练 BERT-base，调优 λ_s, λ_b
- [ ] 评估小模型基线性能

### Phase 2: 路由模块 (Week 3)
- [ ] 实现不确定性分数计算 (熵 + 原型距离)
- [ ] 实现双阈值路由逻辑
- [ ] 在验证集上搜索最优阈值
- [ ] 分析路由分布 (各类请求比例)

### Phase 3: LLM 集成 (Week 4)
- [ ] 设计 few-shot prompt 模板
- [ ] 用 GPT-3.5 自动生成意图定义
- [ ] 实现动态 few-shot 示例���择
- [ ] 集成 GPT-4 API

### Phase 4: 实验与分析 (Week 5-6)
- [ ] 主实验 (5 个数据集 × 8 个对比方法)
- [ ] 消融实验
- [ ] 成本-性能权衡分析
- [ ] 延迟分析 (P50/P95/P99)
- [ ] 错误分析 (case study)

---

## 9. 参考文献

1. Zaera et al. (ACL 2025) - UDRIL: Uncertainty-Driven LLM Routing for OOS Detection
2. Ghriss et al. (2024) - Hybrid Intent Classification: Confidence-Based Routing
3. Feng et al. (2024) - OOS Intent Detection with Collaborative Large and Small LMs
4. Khosla et al. (NeurIPS 2020) - Supervised Contrastive Learning
5. Zhang et al. (2021) - Adaptive Decision Boundary for OOS Detection
