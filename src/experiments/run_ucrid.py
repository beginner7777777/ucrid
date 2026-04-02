"""
UCRID Main Experiment Script
Runs the full 3-stage cascade pipeline on CLINC150 / Banking77 / HINT3.
"""

import os
import sys
import argparse
from typing import Dict, List

import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config
from data.dataset import load_clinc150_data
from models.bert_encoder import BERTIntentEncoder
from inference.intent_metadata import build_intent_metadata
from inference.ucrid_router import UCRIDRouter
from inference.llm_judge import LLMJudge, apply_llm_label_policy
from utils.utils import compute_metrics, print_metrics, save_results, get_device, set_seed


# ---------------------------------------------------------------------------
# Prototype helpers
# ---------------------------------------------------------------------------

def build_mean_prototypes(
    model: BERTIntentEncoder,
    dataloader,
    device: torch.device,
    num_intents: int,
    oos_label: int,
) -> torch.Tensor:
    """
    Compute mean embedding per in-domain intent.
    Returns: [num_intents, hidden_size]
    """
    model.eval()
    sums = torch.zeros(num_intents, model.hidden_size, device=device)
    counts = torch.zeros(num_intents, device=device)

    with torch.no_grad():
        for batch in dataloader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(ids, mask)
            h = out["hidden_states"]  # [B, D]

            for i in range(h.size(0)):
                lbl = labels[i].item()
                if lbl != oos_label:
                    sums[lbl] += h[i]
                    counts[lbl] += 1

    counts = counts.clamp(min=1)
    return sums / counts.unsqueeze(1)  # [num_intents, D]


def collect_model_outputs(
    model: BERTIntentEncoder,
    dataloader,
    device: torch.device,
):
    """Collect logits, embeddings, and labels for a full split."""
    all_logits, all_hidden, all_labels = [], [], []
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model(ids, mask)
            all_logits.append(out["logits"].detach().cpu())
            all_hidden.append(out["hidden_states"].detach().cpu())
            all_labels.append(batch["labels"].detach().cpu())

    return torch.cat(all_logits), torch.cat(all_hidden), torch.cat(all_labels)


# ---------------------------------------------------------------------------
# Threshold search on validation set
# ---------------------------------------------------------------------------

def search_thresholds(
    router: UCRIDRouter,
    model: BERTIntentEncoder,
    val_loader,
    prototypes: torch.Tensor,
    device: torch.device,
    oos_label: int,
    alpha_grid=(0.3, 0.5, 0.7),
    tau_accept_grid=(0.2, 0.3, 0.4),
    tau_reject_grid=(0.7, 0.8, 0.9),
    delta_grid=(1.0,),
    max_oos_f1_drop=0.01,
    fit_temperature=True,
):
    """
    Joint search over alpha / thresholds:
    minimize LLM call rate s.t. OOS F1 drop <= max_oos_f1_drop.
    """
    logits, hidden, labels_t = collect_model_outputs(model, val_loader, device)
    labels = labels_t.numpy()
    protos_cpu = prototypes.cpu()

    if fit_temperature:
        router.fit_temperature(logits, labels_t)

    H = router.compute_entropy(logits).cpu().numpy()
    d = router.compute_d_min(hidden, protos_cpu).cpu().numpy()
    router.calibrate(H, d)

    top1 = router.scale_logits(logits).argmax(dim=-1).numpy()
    baseline_metrics = compute_metrics(top1, labels, oos_label=oos_label)

    best_feasible = None
    best_overall = None

    def better_feasible(candidate, incumbent):
        if incumbent is None:
            return True
        candidate_key = (
            candidate["llm_rate"],
            -candidate["metrics"]["oos_f1"],
            -candidate["metrics"]["accuracy"],
        )
        incumbent_key = (
            incumbent["llm_rate"],
            -incumbent["metrics"]["oos_f1"],
            -incumbent["metrics"]["accuracy"],
        )
        return candidate_key < incumbent_key

    def better_overall(candidate, incumbent):
        if incumbent is None:
            return True
        candidate_key = (
            -candidate["metrics"]["oos_f1"],
            -candidate["metrics"]["accuracy"],
            candidate["llm_rate"],
        )
        incumbent_key = (
            -incumbent["metrics"]["oos_f1"],
            -incumbent["metrics"]["accuracy"],
            incumbent["llm_rate"],
        )
        return candidate_key < incumbent_key

    for alpha in alpha_grid:
        router.alpha = alpha
        for delta in delta_grid:
            router.delta = delta
            for ta in tau_accept_grid:
                for tr in tau_reject_grid:
                    if ta >= tr:
                        continue
                    router.tau_accept = ta
                    router.tau_reject = tr
                    result = router.route(logits, hidden, protos_cpu)
                    decisions = result["decisions"]
                    preds = result["predictions"].clone().cpu().numpy()

                    for i, decision in enumerate(decisions):
                        if decision == "llm":
                            preds[i] = top1[i]

                    metrics = compute_metrics(preds, labels, oos_label=oos_label)
                    llm_rate = decisions.count("llm") / len(decisions)
                    candidate = {
                        "alpha": alpha,
                        "tau_accept": ta,
                        "tau_reject": tr,
                        "delta": delta,
                        "llm_rate": llm_rate,
                        "metrics": metrics,
                        "routing": router.routing_stats(decisions),
                    }

                    if better_overall(candidate, best_overall):
                        best_overall = candidate

                    if metrics["oos_f1"] >= baseline_metrics["oos_f1"] - max_oos_f1_drop:
                        if better_feasible(candidate, best_feasible):
                            best_feasible = candidate

    best = best_feasible or best_overall
    router.alpha = best["alpha"]
    router.tau_accept = best["tau_accept"]
    router.tau_reject = best["tau_reject"]
    router.delta = best["delta"]

    print(
        "Best routing config: "
        f"alpha={best['alpha']}, tau_accept={best['tau_accept']}, "
        f"tau_reject={best['tau_reject']}, delta={best['delta']}, "
        f"temperature={router.temperature:.4f}, LLM rate={best['llm_rate']:.1%}, "
        f"OOS F1={best['metrics']['oos_f1']:.4f}"
    )
    best = {
        **best,
        "temperature": router.temperature,
        "baseline_metrics": baseline_metrics,
        "constraint_satisfied": best_feasible is not None,
    }
    return best


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_stage1(
    model: BERTIntentEncoder,
    dataloader,
    device: torch.device,
    oos_label: int,
):
    """Evaluate Stage 1 top-1 predictions only."""
    logits, _, labels_t = collect_model_outputs(model, dataloader, device)
    preds = logits.argmax(dim=-1).numpy()
    labels = labels_t.numpy()
    return compute_metrics(preds, labels, oos_label=oos_label)

def evaluate_ucrid(
    model: BERTIntentEncoder,
    router: UCRIDRouter,
    test_loader,
    prototypes: torch.Tensor,
    device: torch.device,
    oos_label: int,
    llm_judge: LLMJudge = None,
    intent_names: list = None,
    intent_defs: dict = None,
    train_examples: dict = None,
    oos_pool: list = None,
    intent_name_to_id: dict = None,
    tokenizer=None,
    collect_details: bool = False,
    llm_accept_policy: str = "all",
):
    model.eval()
    all_preds, all_labels = [], []
    all_decisions = []
    latencies = {"small_model": [], "direct_oos": [], "llm": []}
    prediction_details = []

    import time

    def label_to_name(label_id: int) -> str:
        if label_id == oos_label:
            return "OOS"
        if intent_names is not None and 0 <= label_id < len(intent_names):
            return intent_names[label_id]
        return str(label_id)

    for batch in tqdm(test_loader, desc="Evaluating UCRID"):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        t0 = time.time()
        with torch.no_grad():
            out = model(ids, mask)
        logits = out["logits"]
        hidden = out["hidden_states"]

        result = router.route(logits, hidden, prototypes)
        decisions = result["decisions"]
        stage2_preds = result["predictions"].clone()
        preds = stage2_preds.clone()
        llm_result_map: Dict[int, Dict] = {}
        queries = [
            tokenizer.decode(ids[i].cpu(), skip_special_tokens=True).strip()
            for i in range(ids.size(0))
        ] if tokenizer is not None else [""] * ids.size(0)

        llm_indices = [i for i, d in enumerate(decisions) if d == "llm"]
        for i in llm_indices:
            stage2_preds[i] = result["top1_intents"][i]
            preds[i] = result["top1_intents"][i]

        # Stage 3: LLM judge for uncertain samples
        if llm_indices:
            if llm_judge is not None and tokenizer is not None:
                llm_queries = [queries[i] for i in llm_indices]
                topk_names = [
                    [intent_names[idx] for idx in result["topk_intents"][i]
                     if idx < len(intent_names)]
                    for i in llm_indices
                ]
                llm_results = llm_judge.judge_batch(
                    llm_queries, topk_names, intent_defs, train_examples,
                    oos_pool, intent_name_to_id, oos_label
                )
                for j, i in enumerate(llm_indices):
                    llm_result_map[i] = llm_results[j]
                    fallback_label = int(result["top1_intents"][i].item())
                    resolved_label, llm_applied = apply_llm_label_policy(
                        llm_label=llm_results[j]["label"],
                        fallback_label=fallback_label,
                        oos_label=oos_label,
                        policy=llm_accept_policy,
                    )
                    llm_result_map[i]["resolved_label"] = resolved_label
                    llm_result_map[i]["llm_applied"] = llm_applied
                    preds[i] = resolved_label
            else:
                for i in llm_indices:
                    preds[i] = result["top1_intents"][i]

        elapsed = (time.time() - t0) * 1000 / ids.size(0)
        for dec in decisions:
            latencies[dec].append(elapsed)

        all_preds.append(preds.cpu())
        all_labels.append(labels)
        all_decisions.extend(decisions)

        if collect_details:
            for i in range(ids.size(0)):
                llm_result = llm_result_map.get(i, {})
                gold_label = int(labels[i].item())
                stage1_label = int(result["top1_intents"][i].item())
                stage2_label = int(stage2_preds[i].item())
                final_label = int(preds[i].item())
                prediction_details.append({
                    "query": queries[i],
                    "gold_label": gold_label,
                    "gold_name": label_to_name(gold_label),
                    "decision": decisions[i],
                    "stage1_top1_label": stage1_label,
                    "stage1_top1_name": label_to_name(stage1_label),
                    "stage2_proxy_label": stage2_label,
                    "stage2_proxy_name": label_to_name(stage2_label),
                    "final_label": final_label,
                    "final_name": label_to_name(final_label),
                    "uncertainty_score": float(result["scores"][i].item()),
                    "entropy_norm": float(result["entropy_norm"][i].item()),
                    "distance_norm": float(result["distance_norm"][i].item()),
                    "d_min": float(result["d_min"][i].item()),
                    "topk_intents": [
                        label_to_name(idx) for idx in result["topk_intents"][i]
                    ],
                    "topk_probs": [float(p) for p in result["topk_probs"][i]],
                    "llm_intent_name": llm_result.get("intent_name"),
                    "llm_label": llm_result.get("label"),
                    "llm_resolved_label": llm_result.get("resolved_label"),
                    "llm_raw": llm_result.get("raw"),
                    "llm_used": bool(llm_result.get("llm_applied", False)),
                })

    preds_np = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()

    metrics = compute_metrics(preds_np, labels_np, oos_label=oos_label)
    routing = router.routing_stats(all_decisions)

    # Latency stats
    def pct(arr, p):
        return float(np.percentile(arr, p)) if arr else 0.0

    all_lat = [l for v in latencies.values() for l in v]
    metrics["latency_p50"] = pct(all_lat, 50)
    metrics["latency_p95"] = pct(all_lat, 95)
    metrics["latency_p99"] = pct(all_lat, 99)
    metrics["llm_call_rate"] = routing["llm"]
    metrics["routing"] = routing
    metrics["routing_counts"] = {
        "small_model": all_decisions.count("small_model"),
        "direct_oos": all_decisions.count("direct_oos"),
        "llm": all_decisions.count("llm"),
    }
    metrics["llm_enabled"] = llm_judge is not None

    return metrics, prediction_details


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    config = load_config(args.config)
    set_seed(config.get("training.seed", 42))
    device = get_device(args.gpu_id)

    output_dir = os.path.join("outputs", "ucrid", args.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(config.get("model.bert_model"))

    data_dir = config.get("dataset.data_dir")
    train_loader, val_loader, test_loader = load_clinc150_data(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=config.get("dataset.max_seq_length"),
        batch_size=config.get("inference.batch_size", 32),
        num_workers=config.get("training.num_workers", 4),
        oos_label=config.get("dataset.oos_label", 150),
    )

    # Load trained model
    model = BERTIntentEncoder(
        bert_model_name=config.get("model.bert_model"),
        num_labels=config.get("model.num_labels"),
        hidden_size=config.get("model.hidden_size"),
        dropout=config.get("model.dropout"),
    ).to(device)

    default_ckpt_candidates = [
        args.checkpoint,
        os.path.join("outputs", "stage1", "ucrid_stage1", "best_model.pt"),
        os.path.join("checkpoints", "best_model.pt"),
    ]
    ckpt_path = next((path for path in default_ckpt_candidates if path and os.path.exists(path)), None)
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint found. Pass --checkpoint explicitly.")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {ckpt_path}")

    # Build mean prototypes from training set
    oos_label = config.get("dataset.oos_label", 150)
    num_intents = config.get("dataset.num_intents", 150)
    intent_names, intent_defs, train_examples, oos_pool, intent_name_to_id = build_intent_metadata(
        train_loader.dataset.examples,
        num_intents=num_intents,
        oos_label=oos_label,
    )
    prototypes = build_mean_prototypes(model, train_loader, device, num_intents, oos_label)

    # Stage 2: Router
    router = UCRIDRouter(
        tau_accept=config.get("routing.tau_accept", 0.3),
        tau_reject=config.get("routing.tau_reject", 0.8),
        delta=config.get("routing.delta", 1.0),
        alpha=config.get("routing.alpha", 0.5),
        top_k=config.get("routing.top_k_candidates", 3),
        oos_label=oos_label,
        temperature=config.get("routing.temperature", 1.0),
    )

    # Calibrate and search thresholds on val set
    best_thresholds = search_thresholds(
        router=router,
        model=model,
        val_loader=val_loader,
        prototypes=prototypes,
        device=device,
        oos_label=oos_label,
        alpha_grid=tuple(config.get("routing.alpha_grid", [0.3, 0.5, 0.7])),
        tau_accept_grid=tuple(config.get("routing.tau_accept_grid", [0.2, 0.3, 0.4])),
        tau_reject_grid=tuple(config.get("routing.tau_reject_grid", [0.7, 0.8, 0.9])),
        delta_grid=tuple(config.get("routing.delta_grid", [config.get("routing.delta", 1.0)])),
        max_oos_f1_drop=config.get("routing.search_max_oos_f1_drop", 0.01),
        fit_temperature=config.get("routing.calibrate_temperature", True),
    )

    # Stage 3: LLM judge (optional)
    llm_judge = None
    if args.use_llm:
        try:
            llm_backend = config.get("llm_judge.backend", "local")
            llm_model = config.get("llm_judge.model", "local")
            client = None

            if llm_backend == "anthropic":
                import anthropic
                client = anthropic.Anthropic()
            elif llm_backend == "openai":
                import openai
                openai_kwargs = {}
                base_url = config.get("llm_judge.base_url", None)
                api_key = config.get("llm_judge.api_key", None)
                if base_url:
                    openai_kwargs["base_url"] = base_url
                if api_key:
                    openai_kwargs["api_key"] = api_key
                client = openai.OpenAI(**openai_kwargs)

            llm_judge = LLMJudge(
                client=client,
                model=llm_model,
                backend=llm_backend,
                few_shot_k=config.get("llm_judge.few_shot_k", 3),
                oos_examples=config.get("llm_judge.oos_examples", 2),
                max_tokens=config.get("llm_judge.max_tokens", 32),
                temperature=config.get("llm_judge.temperature", 0.0),
                model_path=llm_model if llm_backend == "local" else None,
                shuffle_candidates=config.get("llm_judge.shuffle_candidates", True),
                random_seed=config.get("llm_judge.random_seed", 42),
                local_batch_size=config.get("llm_judge.local_batch_size", 8),
                disable_thinking=config.get("llm_judge.disable_thinking", False),
                openai_extra_body=config.get("llm_judge.openai_extra_body", {}),
            )
            print(f"LLM judge enabled (backend={llm_backend}, model={llm_model}).")
        except Exception as e:
            print(f"LLM judge unavailable: {e}. Running without Stage 3.")

    stage1_metrics = evaluate_stage1(
        model=model,
        dataloader=test_loader,
        device=device,
        oos_label=oos_label,
    )
    llm_accept_policy = config.get("llm_judge.accept_policy", "all")

    stage2_metrics, _ = evaluate_ucrid(
        model=model,
        router=router,
        test_loader=test_loader,
        prototypes=prototypes,
        device=device,
        oos_label=oos_label,
        llm_judge=None,
        intent_names=intent_names,
        intent_defs=intent_defs,
        train_examples=train_examples,
        oos_pool=oos_pool,
        intent_name_to_id=intent_name_to_id,
        tokenizer=tokenizer,
        collect_details=False,
        llm_accept_policy=llm_accept_policy,
    )
    final_metrics, final_prediction_details = evaluate_ucrid(
        model=model,
        router=router,
        test_loader=test_loader,
        prototypes=prototypes,
        device=device,
        oos_label=oos_label,
        llm_judge=llm_judge,
        intent_names=intent_names,
        intent_defs=intent_defs,
        train_examples=train_examples,
        oos_pool=oos_pool,
        intent_name_to_id=intent_name_to_id,
        tokenizer=tokenizer,
        collect_details=config.get("evaluation.save_predictions", True),
        llm_accept_policy=llm_accept_policy,
    )

    print_metrics(stage1_metrics, "Stage 1 Test")
    print_metrics(stage2_metrics, "Stage 1+2 Test")
    print_metrics(final_metrics, "UCRID Test")
    print(f"LLM call rate: {final_metrics['llm_call_rate']:.1%}")
    print(
        f"Latency P50/P95/P99: {final_metrics['latency_p50']:.1f} / "
        f"{final_metrics['latency_p95']:.1f} / {final_metrics['latency_p99']:.1f} ms"
    )

    results = {
        **final_metrics,
        "stage1_metrics": stage1_metrics,
        "stage2_proxy_metrics": stage2_metrics,
        "final_stage": "stage3_llm" if llm_judge is not None else "stage2_proxy",
        "best_thresholds": best_thresholds,
        "router_config": {
            "alpha": router.alpha,
            "tau_accept": router.tau_accept,
            "tau_reject": router.tau_reject,
            "delta": router.delta,
            "temperature": router.temperature,
            "top_k": router.top_k,
        },
        "llm_accept_policy": llm_accept_policy,
    }
    save_results(results, os.path.join(output_dir, "ucrid_results.json"))
    if config.get("evaluation.save_predictions", True):
        save_results(
            {"prediction_details": final_prediction_details},
            os.path.join(output_dir, "ucrid_prediction_details.json"),
        )
    print(f"\nResults saved to {output_dir}/ucrid_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UCRID Evaluation")
    parser.add_argument("--config", default="configs/clinc150_config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to trained model checkpoint")
    parser.add_argument("--exp_name", default="run1")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--use_llm", action="store_true", help="Enable Stage 3 LLM judge")
    args = parser.parse_args()
    main(args)
