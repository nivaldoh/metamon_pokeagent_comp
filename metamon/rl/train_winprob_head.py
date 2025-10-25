#!/usr/bin/env python3
"""
Finetune a WinProbabilityHead on top of a frozen pretrained Metamon/AMAGO policy.

This trains only the new head with a simple BCE loss against the final outcome label
per trajectory, using the parsed replay dataset. The base agent (tstep encoder and
trajectory encoder) remains frozen and provides sequence embeddings.

Label construction: we recommend using a sparse terminal reward function (BinaryReward
or AggressiveShapedReward) so that the final reward indicates win (> 0) or loss (<= 0).
This script finds the terminal step for each sequence via the `dones` mask and uses the
sign of the final reward as the label.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from tqdm import tqdm

import wandb
from metamon.rl.pretrained import get_pretrained_model
from metamon.rl.train import create_offline_dataset
from metamon.interface import get_reward_function, TokenizedObservationSpace, get_observation_space, get_action_space
from metamon.tokenizer import get_tokenizer
from metamon.rl.win_probability_head import MetamonWinProbAgent, WinProbabilityHead
import amago
from metamon.rl.train import WANDB_PROJECT, WANDB_ENTITY


@dataclass
class TrainConfig:
    model_name: str = "Abra"
    checkpoint: Optional[int] = None
    save_dir: str = "./winprob_runs"
    run_name: str = "winprob_abra"
    # data
    parsed_replay_dir: Optional[str] = None
    custom_replay_dir: Optional[str] = None
    custom_replay_sample_weight: float = 0.25
    formats: Optional[list[str]] = None
    reward_name: str = "BinaryReward"  # recommended to keep labels clean
    # optimization
    epochs: int = 5
    steps_per_epoch: int = 10000
    batch_size: int = 12
    grad_accum: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0
    val_steps: int = 2000
    mixed_precision: bool = True
    device: Optional[str] = None
    log: bool = True


def build_label_from_batch(batch: amago.loading.Batch) -> torch.Tensor:
    """Derive per-sequence outcome labels from terminal reward.

    Assumes a reward function where the final reward is strongly positive for a win
    and non-positive for a loss (e.g., BinaryReward or AggressiveShapedReward).
    """
    rews = batch.rews  # [B, L-1, 1]
    dones = batch.dones  # [B, L-1, 1]
    B, Lm1, _ = rews.shape
    dones_b = dones.squeeze(-1).int()  # [B, L-1]
    # last True index; since only the last step is True, argmax finds it
    last_idx = torch.argmax(dones_b, dim=1)  # [B]
    final_reward = rews[torch.arange(B, device=rews.device), last_idx, 0]  # [B]
    y = (final_reward > 0).float().unsqueeze(-1)  # [B, 1]
    return y


def main(cfg: TrainConfig):
    os.makedirs(os.path.join(cfg.save_dir, cfg.run_name, "ckpts"), exist_ok=True)

    # 1) Load pretrained policy (frozen backbone)
    pretrained = get_pretrained_model(cfg.model_name)
    exp = pretrained.initialize_agent(checkpoint=cfg.checkpoint, log=False)
    policy = exp.policy  # unwrapped AMAGO Agent
    device = cfg.device or exp.DEVICE

    # 2) Wrap with WinProb head
    head_wrapper = MetamonWinProbAgent(base_agent=policy)
    # Freeze base agent
    for p in policy.parameters():
        p.requires_grad = False
    policy.eval()

    # Optimizer on head only
    optim = AdamW(head_wrapper.parameters_for_optimization, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 3) Build dataset
    # Use the same obs/action spaces as the pretrained model, but choose a reward for labeling
    obs_space = pretrained.observation_space
    action_space = pretrained.action_space
    reward_fn = get_reward_function(cfg.reward_name)
    amago_dset = create_offline_dataset(
        obs_space=obs_space,
        action_space=action_space,
        reward_function=reward_fn,
        parsed_replay_dir=cfg.parsed_replay_dir,
        custom_replay_dir=cfg.custom_replay_dir,
        custom_replay_sample_weight=cfg.custom_replay_sample_weight,
        formats=cfg.formats,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision and torch.cuda.is_available())

    policy.to(device)
    head_wrapper.to(device)

    if cfg.log:
        wandb.init(
            project=WANDB_PROJECT or "metamon-winprob",
            entity=WANDB_ENTITY,
            name=cfg.run_name,
            config={k: v for k, v in cfg.__dict__.items()},
        )

    def validate() -> dict:
        head_wrapper.head.eval()
        total = 0
        ys = []
        ps = []
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=cfg.mixed_precision and torch.cuda.is_available()):
                pbar_v = tqdm(total=cfg.val_steps, desc="Val", leave=False)
                while total < cfg.val_steps:
                    samples = [amago_dset.sample_random_trajectory() for _ in range(cfg.batch_size)]
                    batch = amago.loading.RLData_pad_collate(samples).to(device)
                    # Use the last pre-terminal timestep for evaluation
                    _, logits, _ = head_wrapper.logits_all_steps_from_batch(batch)
                    prob_all = torch.sigmoid(logits).squeeze(-1)  # [B,L]
                    last_done = torch.argmax(batch.dones.squeeze(-1).int(), dim=1)  # [B]
                    B = last_done.shape[0]
                    p_sel = prob_all[torch.arange(B, device=prob_all.device), last_done].unsqueeze(-1)  # [B,1]
                    y = build_label_from_batch(batch)  # [B,1]
                    ys.append(y.detach().cpu())
                    ps.append(p_sel.detach().cpu())
                    total += y.shape[0]
                    pbar_v.update(y.shape[0])
                pbar_v.close()
        y_all = torch.cat(ys, dim=0).squeeze(-1).numpy()
        p_all = torch.cat(ps, dim=0).squeeze(-1).numpy()

        # Brier score
        brier = float(((p_all - y_all) ** 2).mean())

        # Accuracy at 0.5
        acc = float(((p_all >= 0.5) == (y_all >= 0.5)).mean())

        # ROC AUC via rank statistic (Mann-Whitney U)
        try:
            import numpy as np
            order = np.argsort(p_all)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(p_all)) + 1  # 1-based ranks
            pos = y_all == 1
            n_pos = int(pos.sum())
            n_neg = len(p_all) - n_pos
            if n_pos > 0 and n_neg > 0:
                sum_pos_ranks = ranks[pos].sum()
                auc = float((sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
            else:
                auc = float("nan")
        except Exception:
            auc = float("nan")

        return {"val/brier": brier, "val/acc@0.5": acc, "val/auc": auc}

    steps_total = 0
    best_auc = -1.0
    best_brier = float("inf")
    for epoch in range(cfg.epochs):
        # train only the head
        head_wrapper.head.train()
        running_loss = 0.0
        pbar = tqdm(range(cfg.steps_per_epoch), desc=f"Epoch {epoch+1}/{cfg.epochs}")
        optim.zero_grad(set_to_none=True)
        for step_in_epoch in pbar:
            # sample a batch of RLData and collate to Batch
            samples = [amago_dset.sample_random_trajectory() for _ in range(cfg.batch_size)]
            batch = amago.loading.RLData_pad_collate(samples).to(device)

            # forward
            with torch.cuda.amp.autocast(enabled=cfg.mixed_precision and torch.cuda.is_available()):
                # Per-timestep logits and no-future masking to avoid leakage
                s_rep, logits, valid_mask = head_wrapper.logits_all_steps_from_batch(batch)
                y_seq = build_label_from_batch(batch).unsqueeze(1).expand_as(logits)
                per_ts_loss = F.binary_cross_entropy_with_logits(logits, y_seq, reduction="none")
                mask = valid_mask.float()
                loss = (per_ts_loss * mask).sum() / mask.sum().clamp_min(1.0)

            # backward
            scaler.scale(loss / cfg.grad_accum).backward()

            steps_total += 1
            running_loss += float(loss.detach().item())

            if (step_in_epoch + 1) % cfg.grad_accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            pbar.set_postfix({"loss": running_loss / (step_in_epoch + 1)})

        # save head checkpoint
        ckpt_path = os.path.join(cfg.save_dir, cfg.run_name, "ckpts", f"winprob_head_epoch_{epoch+1}.pt")
        torch.save({
            "head_state_dict": head_wrapper.head.state_dict(),
            "base_model": cfg.model_name,
            "checkpoint": cfg.checkpoint or pretrained.default_checkpoint,
            "config": cfg.__dict__,
        }, ckpt_path)
        print(f"[INFO] Saved {ckpt_path}")

        # validation
        metrics = validate()
        print(f"[VAL] epoch={epoch+1} auc={metrics['val/auc']:.4f} brier={metrics['val/brier']:.4f} acc@0.5={metrics['val/acc@0.5']:.4f}")
        if cfg.log:
            wandb.log({"train/loss": running_loss / max(1, cfg.steps_per_epoch), **metrics, "epoch": epoch+1})

        # track best by AUC, break ties with lower Brier
        is_best = False
        auc = metrics["val/auc"]
        brier = metrics["val/brier"]
        if auc == auc:  # not NaN
            if auc > best_auc or (abs(auc - best_auc) < 1e-6 and brier < best_brier):
                best_auc = auc
                best_brier = brier
                is_best = True
        elif brier < best_brier:
            best_brier = brier
            is_best = True

        if is_best:
            best_path = os.path.join(cfg.save_dir, cfg.run_name, "ckpts", "winprob_head_best.pt")
            torch.save({
                "head_state_dict": head_wrapper.head.state_dict(),
                "base_model": cfg.model_name,
                "checkpoint": cfg.checkpoint or pretrained.default_checkpoint,
                "config": cfg.__dict__,
                "metrics": metrics,
            }, best_path)
            print(f"[INFO] Updated best checkpoint: {best_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train WinProbabilityHead on top of a frozen pretrained model")
    parser.add_argument("--model", default="Abra", choices=["Abra", "SmallRL", "MediumRL", "SyntheticRLV2", "Minikazam"], help="Pretrained model name")
    parser.add_argument("--ckpt", type=int, default=None, help="Checkpoint number to load (defaults to the model default)")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps_per_epoch", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--reward_name", type=str, default="BinaryReward", choices=["BinaryReward", "AggressiveShapedReward", "DefaultShapedReward"])
    parser.add_argument("--parsed_replay_dir", type=str, default=None)
    parser.add_argument("--custom_replay_dir", type=str, default=None)
    parser.add_argument("--custom_replay_sample_weight", type=float, default=0.25)
    parser.add_argument("--formats", type=str, nargs="+", default=None)
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    # default to gen9ou if not provided
    if args.formats is None:
        args.formats = ["gen9ou"]

    cfg = TrainConfig(
        model_name=args.model,
        checkpoint=args.ckpt,
        save_dir=args.save_dir,
        run_name=args.run_name,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        weight_decay=args.weight_decay,
        reward_name=args.reward_name,
        parsed_replay_dir=args.parsed_replay_dir,
        custom_replay_dir=args.custom_replay_dir,
        custom_replay_sample_weight=args.custom_replay_sample_weight,
        formats=args.formats,
        mixed_precision=not args.no_amp,
    )
    main(cfg)
