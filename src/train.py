"""
UCRID Stage 1 Training Script
Trains BERT with L_CE + λ_s·L_SupCon + λ_b·L_Boundary.
Mean prototypes are rebuilt every epoch for boundary loss.
"""

import os
import sys
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config
from data.dataset import load_clinc150_data
from models.bert_encoder import BERTIntentEncoder
from losses.multi_task_loss import MultiTaskLoss
from utils.utils import (
    set_seed, save_checkpoint, compute_metrics, print_metrics,
    save_results, get_device, count_parameters, AverageMeter
)

import numpy as np


def train_epoch_multitask(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    prototype_bank: torch.Tensor = None
):
    """Train one epoch with multi-task loss"""
    model.train()

    losses = AverageMeter()
    loss_ce_meter = AverageMeter()
    loss_con_meter = AverageMeter()
    loss_bound_meter = AverageMeter()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask)
        logits = outputs['logits']
        hidden_states = outputs['hidden_states']

        # Compute multi-task loss
        loss_dict = criterion(
            logits=logits,
            hidden_states=hidden_states,
            labels=labels,
            prototype_bank=prototype_bank
        )

        loss = loss_dict['loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Update metrics
        losses.update(loss.item(), input_ids.size(0))
        loss_ce_meter.update(loss_dict['loss_ce'], input_ids.size(0))
        loss_con_meter.update(loss_dict['loss_contrastive'], input_ids.size(0))
        loss_bound_meter.update(loss_dict['loss_boundary'], input_ids.size(0))

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'ce': f'{loss_ce_meter.avg:.4f}',
            'con': f'{loss_con_meter.avg:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })

    return {
        'loss': losses.avg,
        'loss_ce': loss_ce_meter.avg,
        'loss_contrastive': loss_con_meter.avg,
        'loss_boundary': loss_bound_meter.avg
    }


def build_mean_prototypes(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_intents: int,
    oos_label: int,
) -> torch.Tensor:
    """Compute mean embedding per in-domain intent. Returns [num_intents, 1, hidden_size]."""
    model.eval()
    hidden_size = model.hidden_size
    sums = torch.zeros(num_intents, hidden_size, device=device)
    counts = torch.zeros(num_intents, device=device)

    with torch.no_grad():
        for batch in dataloader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            h = model(ids, mask)['hidden_states']
            for i in range(h.size(0)):
                lbl = labels[i].item()
                if lbl != oos_label:
                    sums[lbl] += h[i]
                    counts[lbl] += 1

    counts = counts.clamp(min=1)
    means = sums / counts.unsqueeze(1)  # [num_intents, hidden_size]
    return means.unsqueeze(1)           # [num_intents, 1, hidden_size]


def main(args):
    """UCRID Stage 1: train BERT with CE + SupCon + Boundary loss."""

    config = load_config(args.config)
    set_seed(config.get('training.seed', 42))
    device = get_device(args.gpu_id)

    output_dir = os.path.join('outputs', 'stage1', args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    config.save(os.path.join(output_dir, 'config.yaml'))

    tokenizer = BertTokenizer.from_pretrained(config.get('model.bert_model'))

    print("\nLoading data...")
    train_loader, val_loader, test_loader = load_clinc150_data(
        data_dir=config.get('dataset.data_dir'),
        tokenizer=tokenizer,
        max_length=config.get('dataset.max_seq_length'),
        batch_size=config.get('training.batch_size'),
        num_workers=config.get('training.num_workers'),
        oos_label=config.get('dataset.oos_label', 150),
    )

    print("\nCreating model...")
    model = BERTIntentEncoder(
        bert_model_name=config.get('model.bert_model'),
        num_labels=config.get('model.num_labels'),
        hidden_size=config.get('model.hidden_size'),
        dropout=config.get('model.dropout')
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = MultiTaskLoss(
        lambda_contrastive=config.get('loss.lambda_contrastive'),
        lambda_boundary=config.get('loss.lambda_boundary'),
        temperature=config.get('loss.temperature'),
        margin=config.get('loss.margin'),
        oos_label=config.get('dataset.oos_label')
    )

    # Use different LRs: higher for classifier head, lower for BERT backbone
    bert_params = list(model.bert.parameters())
    head_params = list(model.classifier.parameters())
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('training.learning_rate'),
        weight_decay=config.get('training.weight_decay')
    )

    num_training_steps = len(train_loader) * config.get('training.num_epochs')
    warmup_steps = int(num_training_steps * config.get('training.warmup_ratio', 0.1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    oos_label = config.get('dataset.oos_label')
    num_intents = config.get('dataset.num_intents')
    lambda_contrastive = config.get('loss.lambda_contrastive')
    lambda_boundary = config.get('loss.lambda_boundary')
    best_val_oos_f1 = 0.0
    best_ckpt_path = os.path.join(output_dir, 'best_model.pt')

    # Phased training schedule (configurable):
    #   Phase 1 (epochs <= CE_ONLY_EPOCHS)        : CE only
    #   Phase 2 (CE_ONLY_EPOCHS < epoch < BOUNDARY_START) : CE + SupCon
    #   Phase 3 (epochs >= BOUNDARY_START)        : CE + SupCon + Boundary
    # Default keeps previous behavior (3/7).
    CE_ONLY_EPOCHS = int(config.get('training.ce_only_epochs', 3))
    BOUNDARY_START = int(config.get('training.boundary_start_epoch', 7))
    if BOUNDARY_START < 1:
        raise ValueError(f"training.boundary_start_epoch must be >= 1, got {BOUNDARY_START}")
    if CE_ONLY_EPOCHS < 0:
        raise ValueError(f"training.ce_only_epochs must be >= 0, got {CE_ONLY_EPOCHS}")
    if BOUNDARY_START <= CE_ONLY_EPOCHS:
        # This would skip the CE+SupCon middle phase entirely. We allow this,
        # but keep behavior explicit in logs.
        print(
            f"Warning: boundary_start_epoch ({BOUNDARY_START}) <= ce_only_epochs ({CE_ONLY_EPOCHS}); "
            "the CE+SupCon phase will be skipped."
        )

    for epoch in range(1, config.get('training.num_epochs') + 1):
        print(f"\nEpoch {epoch}/{config.get('training.num_epochs')}")

        if epoch <= CE_ONLY_EPOCHS:
            criterion.lambda_contrastive = 0.0
            criterion.lambda_boundary    = 0.0
            prototype_bank = None
            phase = "CE-only"
        elif epoch < BOUNDARY_START:
            criterion.lambda_contrastive = lambda_contrastive
            criterion.lambda_boundary    = 0.0
            prototype_bank = None
            phase = "CE+SupCon"
        else:
            criterion.lambda_contrastive = lambda_contrastive
            criterion.lambda_boundary    = lambda_boundary
            prototype_bank = build_mean_prototypes(
                model, train_loader, device, num_intents, oos_label
            )
            phase = "CE+SupCon+Boundary"

        print(f"  Phase: {phase}")
        train_losses = train_epoch_multitask(
            model, train_loader, criterion, optimizer, scheduler,
            device, epoch, prototype_bank
        )

        print(f"  Total: {train_losses['loss']:.4f}  "
              f"CE: {train_losses['loss_ce']:.4f}  "
              f"SupCon: {train_losses['loss_contrastive']:.4f}  "
              f"Boundary: {train_losses['loss_boundary']:.4f}")

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                logits = model(ids, mask)['logits']
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(batch['labels'].numpy())

        val_metrics = compute_metrics(np.array(val_preds), np.array(val_labels), oos_label=oos_label)
        print_metrics(val_metrics, f"Validation (Epoch {epoch})")

        # Save best model: OOS F1 as primary, but only when ID accuracy is reasonable
        # (avoids saving degenerate solutions where model predicts OOS for everything)
        oos_f1 = val_metrics['oos_f1']
        id_acc  = val_metrics['id_accuracy']
        if id_acc >= 0.5 and oos_f1 > best_val_oos_f1:
            best_val_oos_f1 = oos_f1
            save_checkpoint(
                model, optimizer, epoch, train_losses['loss'],
                best_ckpt_path,
                val_metrics=val_metrics
            )

    # Fallback for datasets/splits where validation OOS F1 can stay at 0 and
    # no checkpoint is selected by the strict criterion above.
    if not os.path.exists(best_ckpt_path):
        print("No best checkpoint selected by validation criterion; saving final epoch model as fallback.")
        save_checkpoint(
            model,
            optimizer,
            config.get('training.num_epochs'),
            train_losses['loss'],
            best_ckpt_path,
            val_metrics=val_metrics,
            fallback_checkpoint=True,
        )

    # Final test evaluation
    checkpoint = torch.load(best_ckpt_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            logits = model(ids, mask)['logits']
            test_preds.extend(logits.argmax(dim=1).cpu().numpy())
            test_labels.extend(batch['labels'].numpy())

    test_metrics = compute_metrics(np.array(test_preds), np.array(test_labels), oos_label=oos_label)
    print_metrics(test_metrics, "Test (Stage 1 small model)")

    save_results(
        {'test_metrics': test_metrics, 'best_val_oos_f1': best_val_oos_f1},
        os.path.join(output_dir, 'stage1_results.json')
    )
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UCRID Stage 1 Training')
    parser.add_argument('--config', type=str, default='configs/clinc150_config.yaml')
    parser.add_argument('--exp_name', type=str, default='ucrid_stage1')
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    main(args)
