# UCRID: Uncertainty-Aware Cascade Routing for Intent Detection and Out-of-Scope Detection

## Title Options

- `UCRID: Uncertainty-Aware Cascade Routing for Intent Detection and Out-of-Scope Detection`
- `Small Model First, LLM on Demand: Uncertainty-Aware Cascade Routing for Intent and OOS Detection`
- `Prototype-Distance-Guided Cascade Routing with LLM Arbitration for Efficient OOS Intent Detection`

## Author Block Placeholder

`Anonymous Authors`

## Abstract

Out-of-scope (OOS) detection is a critical requirement for task-oriented dialogue systems, yet existing solutions often face a difficult trade-off between efficiency and robustness. Small intent classifiers are efficient but weak at recognizing hard OOS cases, while large language models (LLMs) are stronger but too costly to invoke for every query. We present **UCRID**, an uncertainty-aware cascade framework that combines a lightweight intent encoder, a dual-signal router, and candidate-constrained LLM arbitration. In Stage 1, a BERT-based encoder is trained with cross-entropy, supervised contrastive learning, and a boundary-aware loss to improve both intent discrimination and OOS separation in embedding space. In Stage 2, the system computes a fused uncertainty score from calibrated softmax entropy and prototype distance, and uses a dual-threshold policy to directly accept confident in-domain samples, directly reject obvious OOS samples, and forward only ambiguous cases. In Stage 3, an LLM performs final judgment on a small routed subset under a constrained candidate set. Experiments on **CLINC150** and **Banking77** show that UCRID achieves strong OOS detection while keeping LLM usage low. Under the best current setting with `Qwen3-8B fixed + oos_only`, UCRID reaches **84.31 OOS F1** on CLINC150 and **94.47 OOS F1** on Banking77 with only **8.1%** and **7.2%** LLM call rates, respectively. Ablation studies further show that Stage-2 routing is essential, prototype distance is a stronger routing signal than entropy, and Stage-3 LLM arbitration remains necessary for the best final performance.

## 1. Introduction

Intent detection systems deployed in real applications must not only recognize known intents accurately, but also reject inputs that fall outside the supported intent inventory. This out-of-scope (OOS) detection problem is especially important in customer service, banking, and task-oriented assistants, where false acceptance of unsupported requests can severely damage user trust.

Despite long-standing progress on OOS detection, the core tension remains unresolved. Lightweight encoders such as BERT offer low latency and low cost, but their OOS detection ability is limited, especially when unknown inputs lie close to known-intent boundaries. Conversely, large language models (LLMs) provide stronger semantic reasoning and open-set judgment, but their computational and monetary cost make full-query deployment impractical.

This work takes the view that these two model families should not be treated as mutually exclusive. Instead, the system should allow a compact encoder to handle the majority of easy samples, while reserving LLM computation only for the minority of ambiguous cases. To make this practical, the routing mechanism must satisfy three requirements: it should identify confident in-domain cases, separate obvious OOS cases, and expose only truly hard boundary samples to the LLM.

Motivated by this, we propose **UCRID**, a three-stage cascade framework for joint intent detection and OOS detection. First, we train a BERT-based encoder with a multi-objective loss combining cross-entropy, supervised contrastive learning, and a boundary-aware objective that explicitly pushes OOS samples away from in-domain intent prototypes. Second, we build a dual-signal uncertainty router that fuses calibrated entropy and prototype distance, and makes a three-way decision: direct in-domain acceptance, direct OOS rejection, or escalation to an LLM. Third, we constrain the LLM to a small candidate set derived from the small model, making final arbitration both safer and cheaper.

The proposed design is driven by practical deployment concerns rather than pure full-model accuracy. In UCRID, the objective of routing is not to replace the LLM completely, but to minimize unnecessary LLM calls while preserving or improving OOS detection quality. This leads to a system that is both interpretable and operationally controllable.

Our main contributions are as follows:

1. We propose a three-stage cascade framework that unifies lightweight intent classification, uncertainty-aware routing, and candidate-constrained LLM arbitration for efficient OOS detection.
2. We introduce a dual-signal routing mechanism that combines calibrated entropy and prototype distance, together with a dual-threshold policy that separates confident in-domain, obvious OOS, and ambiguous samples.
3. We show through controlled ablations on CLINC150 and Banking77 that prototype distance is the dominant routing signal, while Stage-3 LLM arbitration remains essential for the best overall performance.
4. We provide a practical empirical study of open-source LLM backends, showing that `Qwen3-8B fixed` yields the best performance within the current UCRID pipeline on both datasets.

## 2. Related Work

### 2.1 OOS Detection with Discriminative Intent Encoders

Early OOS detection methods typically rely on discriminative confidence estimation, such as maximum softmax probability or learned decision boundaries. These approaches are efficient and easy to deploy, but they often suffer when OOS samples are semantically close to known intents. In practice, confidence alone does not always provide a stable criterion for distinguishing hard in-domain cases from boundary OOS examples.

### 2.2 Representation Learning for OOS Detection

More recent work improves OOS detection by shaping the embedding space directly. Supervised contrastive learning pulls same-intent samples together and separates different intents, producing more compact in-domain clusters. Prototype-based methods extend this intuition by introducing class centers or sub-prototypes that explicitly encode intent geometry. These methods are especially attractive because they provide a natural distance-based signal for open-set recognition.

### 2.3 LLMs for Intent and OOS Detection

LLMs have shown strong zero-shot and few-shot capabilities for intent understanding, but full-query LLM inference remains expensive. Existing studies also show that general-purpose LLMs can be unstable under strict OOS labeling rules, especially when output format and label constraints are not carefully controlled. This motivates hybrid systems in which a small model handles easy cases and the LLM serves as a fallback judge.

### 2.4 Positioning of This Work

UCRID combines ideas from representation learning, prototype-based open-set detection, and hybrid small-model/LLM inference. Relative to pure small-model methods, it introduces a practical cascade mechanism that escalates only uncertain queries. Relative to pure LLM methods, it significantly reduces LLM dependence. Relative to prior prototype methods, it uses prototype distance not only for training supervision but also as an explicit routing signal at inference time.

## 3. Method

### 3.1 Problem Definition

Let $\mathcal{Y}_{ID}=\{1,\dots,C\}$ denote the set of in-domain intent labels, and let $y=\text{OOS}$ denote the out-of-scope label. Given an input utterance $u$, the system must either predict one of the known intents in $\mathcal{Y}_{ID}$ or reject the input as OOS. The goal is to maximize both in-domain intent accuracy and OOS detection performance while minimizing expensive LLM usage.

### 3.2 Overview of UCRID

UCRID consists of three stages:

1. **Stage 1: Small-model intent encoder.** A BERT-based encoder outputs both class logits and a dense utterance embedding.
2. **Stage 2: Uncertainty-aware router.** A dual-signal router combines calibrated entropy and prototype distance to assign each sample to one of three routes: direct in-domain acceptance, direct OOS rejection, or LLM arbitration.
3. **Stage 3: Candidate-constrained LLM judge.** The LLM receives only ambiguous samples and predicts either one candidate intent or OOS under a restricted label space.

This decomposition lets the lightweight encoder handle the majority of easy cases, while the LLM focuses only on the difficult residual region.

### 3.3 Stage 1: Multi-Objective Intent Encoder

We use a BERT-based encoder to produce a hidden representation $h_u$ and class logits $z_u$ for each input. The encoder is trained with a multi-task objective:

\[
\mathcal{L} = \mathcal{L}_{CE} + \lambda_s \mathcal{L}_{SupCon} + \lambda_b \mathcal{L}_{Boundary}.
\]

The cross-entropy term $\mathcal{L}_{CE}$ ensures basic in-domain classification performance. The supervised contrastive term $\mathcal{L}_{SupCon}$ improves representation compactness by pulling same-intent samples together and pushing different-intent samples apart. The boundary-aware term $\mathcal{L}_{Boundary}$ uses prototype distance to explicitly enlarge the gap between OOS samples and in-domain intent prototypes.

For each in-domain class $c$, we construct a mean prototype:

\[
p_c = \frac{1}{N_c} \sum_{i:y_i=c} h_i.
\]

For an OOS training sample $x$, the boundary loss is defined as:

\[
\mathcal{L}_{Boundary} = \max(0, \Delta - d(h_x, p_{c^*})),
\]

where $c^* = \arg\min_c d(h_x, p_c)$ and $d(\cdot,\cdot)$ is Euclidean distance. This objective encourages OOS samples to remain at least margin $\Delta$ away from the nearest in-domain prototype.

To stabilize optimization, training follows a phased schedule. Early epochs use cross-entropy only, middle epochs add supervised contrastive loss, and later epochs activate the boundary-aware term after the encoder has already learned a reasonable intent structure.

### 3.4 Stage 2: Dual-Signal Uncertainty Router

For an input $u$, Stage 2 computes two complementary uncertainty signals:

\[
H(u) = - \sum_y p(y|u)\log p(y|u),
\]

\[
d_{min}(u) = \min_c \|h_u - p_c\|_2.
\]

The first term measures prediction uncertainty from the calibrated softmax distribution, while the second captures geometric deviation from known intent regions. We apply temperature scaling to logits and min-max normalization to both signals on the validation set:

\[
H_{norm}(u) = \frac{H(u)-H_{min}}{H_{max}-H_{min}},
\qquad
d_{norm}(u) = \frac{d_{min}(u)-d_{min}^{global}}{d_{max}^{global}-d_{min}^{global}}.
\]

The final uncertainty score is:

\[
s(u) = \alpha H_{norm}(u) + (1-\alpha)d_{norm}(u).
\]

Routing uses a dual-threshold policy:

- If $s(u)\le \tau_{accept}$, the system directly accepts the Stage-1 top-1 intent.
- If $s(u)\ge \tau_{reject}$ and $d_{min}(u)>\Delta$, the system directly predicts OOS.
- Otherwise, the sample is forwarded to the LLM.

Unlike a single-threshold design, this policy creates three semantically meaningful regions: a confident in-domain region, an obvious OOS region, and an ambiguity band that requires higher-capacity reasoning.

### 3.5 Stage 3: Candidate-Constrained LLM Arbitration

Samples routed to Stage 3 are judged by an LLM under a restricted candidate set. Instead of asking the LLM to classify over the full intent inventory, we pass the top-$k$ candidate intents from the small model together with intent names, intent descriptions, and few-shot examples. The LLM is instructed to return either one of the provided candidates or `OOS`.

This design offers two benefits. First, it reduces the search space of the LLM and therefore improves format stability and label consistency. Second, it makes the overall cascade more efficient by preserving the small model as the dominant component and using the LLM only as an ambiguity resolver.

In the current best setting, we use an `oos_only` acceptance policy, meaning that the LLM is primarily allowed to overturn the cascade result when it provides a valid OOS judgment. This conservative integration policy reduces the risk of noisy intent substitutions.

### 3.6 Validation-Time Search Objective

The router parameters $\alpha$, $\tau_{accept}$, $\tau_{reject}$, and $\Delta$ are selected on the validation set. The search objective is not to maximize Stage-2 OOS F1 alone. Instead, the system minimizes LLM call rate subject to a constraint on acceptable OOS performance degradation. This is consistent with the deployment goal of keeping LLM usage low while preserving strong end-to-end behavior.

## 4. Experimental Setup

### 4.1 Datasets

We evaluate UCRID on three datasets:

- **CLINC150**: a widely used intent detection benchmark with 150 in-domain intents and an explicit OOS category.
- **Banking77**: a fine-grained banking-domain intent dataset, adapted here for OOS detection under the project’s current data pipeline.
- **HINT3-v2**: a practical industrial intent benchmark with three business-specific subsets, namely `curekart`, `sofmattress`, and `powerplay11`. Since the original release does not directly provide the same split structure required by our cascade threshold search, we convert it into the UCRID JSON format and treat `NO_NODES_DETECTED` as the OOS label.

CLINC150 serves as the main multi-domain benchmark, Banking77 tests whether the method generalizes to finer-grained domain-specific intent boundaries, and HINT3 provides an additional domain-transfer evaluation under a more deployment-oriented data condition.

### 4.2 Implementation Summary

The main experiment pipeline is implemented in [run_ucrid.py](/mnt/data3/wzc/llm_oos_detection/src/experiments/run_ucrid.py). Stage-1 training is implemented in [train.py](/mnt/data3/wzc/llm_oos_detection/src/train.py). The Stage-2 router is implemented in [ucrid_router.py](/mnt/data3/wzc/llm_oos_detection/src/inference/ucrid_router.py), and Stage-3 arbitration is implemented in [llm_judge.py](/mnt/data3/wzc/llm_oos_detection/src/inference/llm_judge.py).

The current best-performing open-source LLM configuration is `Qwen3-8B fixed + oos_only`. We also evaluate `Qwen2-7B` and `Mixtral-8x7B-v0.1` as alternative Stage-3 backends.

### 4.3 Evaluation Metrics

We report:

- Overall accuracy
- In-domain accuracy
- OOS precision
- OOS recall
- OOS F1
- LLM call rate

For deployment analysis, latency and LLM usage are also considered when available.

### 4.4 Baselines and Comparison Scope

This paper distinguishes between two kinds of comparison:

1. **Controlled comparisons under our own pipeline**, including Stage-1-only, Stage-1+2, signal-level ablations, and LLM backend variants. These are directly comparable.
2. **Recent literature under heterogeneous protocols**, which are discussed for context only. Because many prior papers use different known-intent split ratios, datasets, or metrics, direct numerical comparison is not statistically fair.

## 5. Main Results

### 5.1 Best End-to-End Results

Under the best current setting (`Qwen3-8B fixed + oos_only`), UCRID achieves the following end-to-end results:

| Dataset | Accuracy | ID Accuracy | OOS Precision | OOS Recall | OOS F1 | LLM Call Rate |
|---|---:|---:|---:|---:|---:|---:|
| CLINC150 | 92.40 | 95.91 | 93.76 | 76.60 | 84.31 | 8.1 |
| Banking77 | 92.65 | 92.44 | 94.95 | 94.00 | 94.47 | 7.2 |

These results show that the system can obtain strong OOS detection while invoking the LLM for only a small minority of samples.

### 5.2 Comparison of Stage-3 LLM Backends

We further compare three open-source LLM backends within the same UCRID framework.

#### CLINC150

| LLM Backend | Accuracy | ID Accuracy | OOS Precision | OOS Recall | OOS F1 | LLM Call Rate |
|---|---:|---:|---:|---:|---:|---:|
| Early open-source 7B baseline (current project record, denoted as Qwen2-7B) | 90.51 | 95.96 | 94.29 | 66.00 | 77.65 | 8.1 |
| Qwen3-8B fixed | **92.40** | **95.91** | 93.76 | **76.60** | **84.31** | 8.1 |
| Mixtral-8x7B-v0.1 ctxsafe | 91.02 | 95.91 | 93.62 | 69.00 | 79.45 | 8.1 |

#### Banking77

| LLM Backend | Accuracy | ID Accuracy | OOS Precision | OOS Recall | OOS F1 | LLM Call Rate |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2-7B baseline | 92.60 | 92.37 | 94.00 | 94.00 | 94.00 | 7.2 |
| Qwen3-8B fixed | **92.65** | 92.44 | **94.95** | **94.00** | **94.47** | 7.2 |
| Mixtral-8x7B-v0.1 ctxsafe | 91.76 | **92.47** | 95.00 | 87.40 | 91.04 | 7.2 |

The results indicate that `Qwen3-8B fixed` is the most reliable Stage-3 backend in the current system. On CLINC150, the gap is especially pronounced, suggesting that this dataset is more sensitive to output stability, OOS formatting consistency, and constrained label following. On Banking77, the gap between `Qwen2-7B` and `Qwen3-8B fixed` is small, while Mixtral remains weaker mainly because of lower OOS recall.

### 5.3 Additional Generalization Results on HINT3-v2

To further examine whether the proposed cascade transfers beyond the two main benchmarks, we evaluate the same `Qwen3-8B + oos_only` pipeline on the three HINT3-v2 business subsets. We report the full end-to-end UCRID results below.

| HINT3-v2 Subset | Accuracy | ID Accuracy | OOS Precision | OOS Recall | OOS F1 | LLM Call Rate |
|---|---:|---:|---:|---:|---:|---:|
| curekart | 67.09 | 47.14 | 66.36 | 84.27 | 74.25 | 25.47 |
| sofmattress | 49.21 | 24.26 | 44.03 | 93.04 | 59.78 | 18.61 |
| powerplay11 | 65.65 | 23.48 | 79.93 | 84.97 | 82.37 | 32.44 |
| Macro average | 60.65 | 31.63 | 63.44 | 87.43 | 72.13 | 25.51 |

These results reveal a noticeably different operating regime from CLINC150 and Banking77. On HINT3-v2, UCRID still preserves strong OOS recall across all three subsets, but in-domain accuracy drops substantially, especially on `sofmattress` and `powerplay11`. This pattern suggests that the HINT3 setting is dominated by a much harsher open-set boundary: many samples are easier to reject as unsupported than to map precisely to the correct in-domain intent. The router therefore behaves more conservatively, sending a larger fraction of traffic to direct OOS rejection and maintaining a moderate LLM call rate. From a paper perspective, this is still a useful result: it shows that the proposed cascade retains robust OOS sensitivity under industrial-style domain shift, but it also highlights that cross-domain intent granularity remains the main bottleneck once the system moves beyond cleaner benchmark distributions.

## 6. Ablation Study

To validate the internal design of UCRID, we perform structured ablations on the current main pipeline. The goal is to isolate the contribution of Stage 2, the relative importance of entropy and prototype distance, and the additional value of Stage-3 LLM arbitration.

### 6.1 CLINC150

| Method | Acc. | ID Acc. | OOS Prec. | OOS Rec. | OOS F1 | LLM Rate |
|---|---:|---:|---:|---:|---:|---:|
| Stage1 only | 86.35 | 96.07 | 95.73 | 42.60 | 58.96 | - |
| Stage1+2 (full router) | 89.55 | 95.96 | 93.82 | 60.70 | 73.71 | 8.1 |
| w/o distance (entropy-only) | 89.11 | 95.98 | 94.33 | 58.20 | 71.99 | 5.9 |
| w/o entropy (distance-only) | 91.07 | 95.76 | 91.50 | 70.00 | 79.32 | 9.6 |
| Full UCRID | **92.40** | 95.91 | 93.76 | **76.60** | **84.31** | 8.1 |

### 6.2 Banking77

| Method | Acc. | ID Acc. | OOS Prec. | OOS Rec. | OOS F1 | LLM Rate |
|---|---:|---:|---:|---:|---:|---:|
| Stage1 only | 79.78 | **92.73** | 0.00 | 0.00 | 0.00 | - |
| Stage1+2 (full router) | 89.94 | 92.47 | 95.38 | 74.40 | 83.60 | 7.2 |
| w/o distance (entropy-only) | 89.61 | 92.53 | **95.72** | 71.60 | 81.92 | 6.2 |
| w/o entropy (distance-only) | 91.17 | 92.34 | 91.50 | 84.00 | 87.59 | 11.9 |
| Full UCRID | **92.65** | 92.44 | 94.95 | **94.00** | **94.47** | 7.2 |

### 6.3 Ablation Analysis

The ablation results support three consistent conclusions.

First, Stage-2 routing is indispensable. On CLINC150, OOS F1 rises from 58.96 to 73.71 when moving from Stage1 only to Stage1+2. On Banking77, the gain is much larger, with OOS F1 improving from 0.00 to 83.60. This shows that the router is not a minor optimization but a core enabler of practical OOS detection.

Second, prototype distance is the stronger routing signal. On both datasets, the distance-only variant outperforms the entropy-only variant in OOS F1. This suggests that hard OOS samples are better characterized as geometrically distant from all known intent prototypes, whereas entropy captures only classifier hesitation and misses part of the structural information.

Third, Stage-3 LLM arbitration remains necessary. Even though distance-based routing is already strong, full UCRID still yields the best final performance on both datasets. This demonstrates that Stage 2 and Stage 3 are complementary: the router isolates hard cases efficiently, while the LLM resolves the residual ambiguity.

## 7. Additional Controlled Comparisons Under the Same Project

Before the full UCRID cascade was finalized, earlier controlled experiments on the same project also demonstrated the effectiveness of contrastive learning and boundary-aware prototype modeling on CLINC150:

| Experiment | Method | OOS F1 | OOS Recall | OOS Precision | ID Accuracy |
|---|---|---:|---:|---:|---:|
| E1 | MaxSoftmax BERT baseline | 73.55 | 61.30 | 93.80 | 95.51 |
| E2 | E1 + SupCon | 76.58 | 67.70 | 86.40 | 95.36 |
| E4 | E2 + Prototype Bank + Boundary Loss | **84.67** | **92.50** | 78.20 | 92.80 |

These results are useful because they isolate the contribution of representation shaping under a fixed small-model evaluation setup. In particular, the gain from E2 to E4 shows that explicit prototype-aware boundary modeling can substantially improve OOS recall.

## 8. Discussion

### 8.1 Why Prototype Distance Matters More than Entropy

Entropy reflects uncertainty in the classifier’s output distribution, but it does not directly describe where the sample lies in feature space. A sample may receive a peaked but still incorrect intent distribution if it falls just outside the learned decision region of a nearby intent. Prototype distance, by contrast, measures whether the sample is actually embedded near any known intent cluster. This geometric property is more closely aligned with the open-set nature of OOS detection.

### 8.2 Why the Full Dual-Signal Router Is Not Always the Best Stage-2 Variant

In our ablation, the distance-only router sometimes outperforms the fused router at the Stage-2 level. This does not invalidate the full design. The current routing parameters are selected by validation search under a joint objective that also considers call budget, rather than maximizing Stage-2 OOS F1 alone. As a result, the fused router is better interpreted as a deployment-oriented trade-off mechanism, while the end-to-end optimum is achieved after Stage-3 arbitration.

### 8.3 Why Qwen3-8B Fixed Outperforms the Earlier Qwen3-8B Setup

The project’s best result is obtained not by the original `Qwen3-8B` run, but by the `Qwen3-8B fixed` configuration. This is an important engineering observation rather than a contradiction. The fixed variant improved output controllability, parsing robustness, and prompt-side constraints, which directly increased the fraction of valid Stage-3 judgments accepted by the cascade. In other words, the gain did not come solely from raw model capability, but from better alignment between the model, the prompt protocol, and the system acceptance policy.

## 9. Limitations

This work has several limitations.

First, some comparisons to recent literature are necessarily contextual rather than directly numerical, because prior papers often use different known-intent splits, datasets, or evaluation metrics. A broader head-to-head benchmark under a unified protocol remains future work.

Second, the current study focuses on open-source LLM backends under the existing project infrastructure. While this is useful for practical deployment, more extensive analysis across larger proprietary LLMs would strengthen the generality of the conclusions.

Third, Banking77 requires project-specific adaptation for OOS evaluation. Although the results are informative, future work should further validate the method on additional naturally OOS-rich benchmarks and real industrial logs.

Fourth, the current system uses a relatively simple prototype definition based on class means. It would be valuable to study richer prototype structures, dynamic prototype refinement, or learned class manifolds in future extensions.

## 10. Conclusion

We introduced UCRID, a three-stage cascade framework for intent detection and OOS detection that combines a lightweight encoder, uncertainty-aware routing, and candidate-constrained LLM arbitration. The method is motivated by a practical objective: preserve the efficiency of small models while selectively exploiting the semantic strength of LLMs only where needed.

Experiments on CLINC150 and Banking77 show that this design is effective. UCRID achieves strong OOS performance with low LLM call rates, and ablation results confirm that Stage-2 routing is essential, prototype distance is the dominant routing signal, and Stage-3 LLM arbitration provides the final performance gains. Together, these findings support the broader view that efficient OOS detection should be treated not as a choice between small models and LLMs, but as a coordinated cascade in which both serve distinct and complementary roles.

## 11. References Draft

This project already compiled a contextual survey of recent related methods in [recent_methods_comparison.md](/mnt/data3/wzc/llm_oos_detection/outputs/recent_methods_comparison.md). A draft bibliography section for the final paper can start from the following entries and should be cleaned into the target venue format:

1. Song et al. Continual Generalized Intent Discovery. Findings of EMNLP, 2023.
2. Song et al. Evaluating ChatGPT for Intent Detection. EMNLP, 2023.
3. Zawbaa et al. DETER: Dual Encoder with Threshold Re-Classification for OOS Detection. LREC-COLING, 2024.
4. Gautam et al. SCOOS: Class Name Guided OOS Detection without OOS Training Data. Findings of EMNLP, 2024.
5. Li et al. Hard Negative Augmentation for OOS Detection with ChatGPT. LREC-COLING, 2024.
6. Zhang et al. RAP: Relation-Aware Prototypes for New Intent Discovery. LREC-COLING, 2024.
7. Castillo-López et al. Enhancing OOS Detection in Multi-party Conversations with LLM and Label Semantic Representations. SIGDIAL, 2025.

## 12. Suggested Submission Positioning

### Recommended Conference Targets

- ACL / EMNLP / COLING if the paper is framed as a practical hybrid NLU system for intent and OOS detection.
- Findings of ACL / EMNLP if you want a slightly lower-risk but still strong NLP venue.
- Information Processing & Management or Expert Systems with Applications if you prefer a journal route emphasizing deployable NLP systems.

### Positioning Advice

- Emphasize the **system contribution**: small model first, LLM on demand.
- Emphasize the **routing contribution**: dual-signal uncertainty with direct OOS rejection.
- Emphasize the **engineering contribution**: constrained Stage-3 arbitration and low LLM call rate.
- Be careful not to oversell cross-paper SOTA unless the protocols are truly aligned.

## 13. Source Files Used for This Draft

- Full logic note: [paper_ucrid_full_logic_20260321.md](/mnt/data3/wzc/llm_oos_detection/outputs/paper_ucrid_full_logic_20260321.md)
- Ablation summary: [ucrid_ablation_paper_ready_20260324.md](/mnt/data3/wzc/llm_oos_detection/outputs/ucrid_ablation_paper_ready_20260324.md)
- LLM comparison: [model_comparison_clinc150_banking77_qwen2_qwen3_mixtral_20260321.md](/mnt/data3/wzc/llm_oos_detection/outputs/model_comparison_clinc150_banking77_qwen2_qwen3_mixtral_20260321.md)
- Earlier controlled comparison: [updated_comparison_table.md](/mnt/data3/wzc/llm_oos_detection/outputs/updated_comparison_table.md)
