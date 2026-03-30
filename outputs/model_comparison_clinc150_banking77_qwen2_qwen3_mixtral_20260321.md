# CLINC150 / Banking77 开源 LLM 对比分析

## 口径说明

- 本文对比 3 组开源大模型结果：`Qwen2-7B`、`Qwen3-8B`、`Mixtral-8x7B-v0.1`。
- `Qwen3-8B` 采用当前效果最好的 fixed 版结果。
- `Mixtral-8x7B-v0.1` 采用 `ctxsafe` 版结果，即为适配 512 上下文窗口后缩短 Stage3 prompt 的版本。
- `Qwen2-7B` 这一列沿用你当前项目里的早期开源 vLLM 基线口径：
  - Banking77 直接对应 `rule_v4`。
  - CLINC150 直接对应 2026-03-19 的早期开源基线结果；该旧汇总里写作 `LLaMA`，因此这一列应视为“早期开源 7B 基线”，如果你后续确认其确实是 Qwen2-7B，可直接把标题替换为 Qwen2-7B。

## 1. 总体结论

- 两个数据集上，当前最优都是 `Qwen3-8B fixed`。
- `Mixtral-8x7B-v0.1` 在两个数据集上都能明显提升 Stage2，但都没有超过 `Qwen3-8B fixed`。
- `Mixtral` ��主要问题不是完全不可用，而是“效果不占优，延迟明显偏高”。
- `Banking77` 上三模型差距比 `CLINC150` 小，但 `Mixtral` 仍然在效果和延迟上都落后于 `Qwen3-8B fixed`。
- `CLINC150` 上模型差异更明显，说明该数据集对 Stage3 的输出格式稳定性、OOS 判定边界、few-shot 质量更敏感。

## 2. CLINC150 对比

| 模型 | Accuracy | ID Accuracy | OOS Precision | OOS Recall | OOS F1 | P95 延迟 | LLM 调用率 | `llm_used` / 路由样本 |
|------|----------|-------------|---------------|------------|--------|----------|------------|----------------------|
| 早期开源 7B 基线（按当前口径记为 Qwen2-7B） | 90.51% | 95.96% | 94.29% | 66.00% | 0.7765 | 211.2 ms | 8.1% | 133 / 443 |
| Qwen3-8B fixed | **92.40%** | **95.91%** | 93.76% | **76.60%** | **0.8431** | **117.0 ms** | 8.1% | **305 / 443** |
| Mixtral-8x7B-v0.1 ctxsafe | 91.02% | 95.91% | 93.62% | 69.00% | 0.7945 | 2593.5 ms | 8.1% | 179 / 443 |

相对 Stage1+2（OOS F1 = `0.7371`）：

- 早期开源 7B 基线提升 `+0.0394`
- Qwen3-8B fixed 提升 `+0.1061`
- Mixtral-8x7B-v0.1 提升 `+0.0574`

对 CLINC150 的解释：

- `Qwen3-8B fixed` 优势最明显，核心不只是最终 F1 更高，而是它在 `oos_only` 策略下真正被接纳的 OOS 判定更多，`llm_used=305`，远高于另外两者。
- 这说明 Qwen3 fixed 版更容易稳定地产出可解析、可接纳、且符合 `OOS` 标签的输出。
- `Mixtral` 的 `OOS Precision` 并不差，但 `OOS Recall` 明显低于 Qwen3 fixed，说明它更保守，很多该判 OOS 的样本没有被成功翻转。
- `Mixtral` 最大短板是延迟，`P95` 从 Qwen3 fixed 的 `117 ms` 上升到 `2593 ms`，已经不是同一量级。
- CLINC150 上 Stage3 的收益高度依赖“模型能否稳定输出严格格式 + 清晰 OOS 决策”，这一点上 Qwen3 fixed 明显最强。

## 3. Banking77 对比

| 模型 | Accuracy | ID Accuracy | OOS Precision | OOS Recall | OOS F1 | P95 延迟 | LLM 调用率 | `llm_used` / 路由样本 |
|------|----------|-------------|---------------|------------|--------|----------|------------|----------------------|
| Qwen2-7B（rule_v4 / vLLM 基线） | 92.60% | 92.37% | 94.00% | 94.00% | 0.9400 | **37.7 ms** | 7.2% | **110 / 258** |
| Qwen3-8B fixed | **92.65%** | 92.44% | 94.95% | **94.00%** | **0.9447** | 39.2 ms | 7.2% | 105 / 258 |
| Mixtral-8x7B-v0.1 ctxsafe | 91.76% | **92.47%** | **95.00%** | 87.40% | 0.9104 | 957.1 ms | 7.2% | 70 / 258 |

相对 Stage1+2（OOS F1 = `0.8360`）：

- Qwen2-7B 基线提升 `+0.1040`
- Qwen3-8B fixed 提升 `+0.1088`
- Mixtral-8x7B-v0.1 提升 `+0.0745`

对 Banking77 的解释：

- Banking77 上 `Qwen2-7B` 和 `Qwen3-8B fixed` 非常接近，`Qwen3-8B fixed` 只小幅领先。
- 这说明 Banking77 的 OOS 判定边界更规则，较强的开源 7B 模型基本都能吃到 Stage3 收益。
- `Mixtral` 这里依旧不是 precision 问题，而是 recall 问题。它把 `OOS Precision` 做到了 `95.00%`，但 `OOS Recall` 掉到 `87.40%`，最终拖累 OOS F1。
- 从 `llm_used` 看，Mixtral 只有 `70 / 258` 被真正接纳，而 Qwen2/Qwen3 都在 `100+`，说明它在 `oos_only` 策略下被采纳的有效 OOS 输出更少。
- 延迟差距仍然很大。`Qwen2/Qwen3` 的 `P95` 都在 `40 ms` 左右，而 Mixtral 接近 `1 s`。

## 4. 为什么 Qwen3-8B 更好

- 第一，`Qwen3-8B fixed` 的输出可控性更强。你已经对它做过关闭 thinking、增强解析、扩大 `max_tokens` 的修复，这些修复直接提高了 Stage3 的可用率。
- 第二，`oos_only` 策略下，真正重要的不是“模型是否聪明”，而是“模型是否能稳定给出可接纳的 OOS 结论”。Qwen3 在这点上表现最好。
- 第三，Mixtral 当前为了能在 4 张 3090 上稳定跑通，被迫使用了 `ctxsafe` 版本：`top_k=2`、`few_shot_k=1`、`oos_examples=1`。这会削弱 Stage3 的判别信息量。
- 第四，Mixtral 还需要依赖 `chat.completions -> completions` fallback，这本身就说明它和当前推理接口、prompt 形态的适配度不如 Qwen3。

## 5. 工程建议

- 如果目标是当前这套 UCRID 流程下的最佳综合效果，优先使用 `Qwen3-8B fixed`。
- 如果目标是低延迟且效果接近最优，Banking77 上 `Qwen2-7B / rule_v4` 仍然很有竞争力。
- 如果要继续提升 `Mixtral`，重点不是继续直接重跑，而是先解决 prompt 容量和格式控制问题，否则它大概率仍会处于“成本更高、收益更小”的状态。
- 如果后续要做更公平的三模型对比，建议统一三件事：
  - 统一 prompt 长度预算
  - 统一 `top_k_candidates`
  - 统一 `few_shot_k / oos_examples`

## 6. 本文使用的结果文件

- CLINC150 早期开源 7B 基线：
  [ucrid_results.json](/mnt/data3/wzc/llm_oos_detection/outputs/ucrid/clinc150_rerun_20260319_104000_stage3_oos_only/ucrid_results.json)
- CLINC150 Qwen3-8B fixed：
  [ucrid_results.json](/mnt/data3/wzc/llm_oos_detection/outputs/ucrid/clinc150_rerun_20260320_1510_oos_only_warnclean_qwen3_8888_fixed/ucrid_results.json)
- CLINC150 Mixtral-8x7B-v0.1：
  [ucrid_results.json](/mnt/data3/wzc/llm_oos_detection/outputs/ucrid/clinc150_rerun_20260321_0530_oos_only_warnclean_mixtral_8010_ctxsafe/ucrid_results.json)
- Banking77 Qwen2-7B / rule_v4：
  [ucrid_results.json](/mnt/data3/wzc/llm_oos_detection/outputs/ucrid/banking77_rerun_20260320_1406_oos_only_warnclean_vllm/ucrid_results.json)
- Banking77 Qwen3-8B fixed：
  [ucrid_results.json](/mnt/data3/wzc/llm_oos_detection/outputs/ucrid/banking77_rerun_20260320_1458_oos_only_warnclean_qwen3_8888_fixed/ucrid_results.json)
- Banking77 Mixtral-8x7B-v0.1：
  [ucrid_results.json](/mnt/data3/wzc/llm_oos_detection/outputs/ucrid/banking77_rerun_20260321_0637_oos_only_warnclean_mixtral_8010_ctxsafe/ucrid_results.json)
