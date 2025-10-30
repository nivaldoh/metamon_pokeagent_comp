#!/usr/bin/env python3
"""
Train a timestep-only WinProbability head on top of a frozen pretrained model.

This finetunes a small MLP head directly on the timestep encoder outputs and
does not use the trajectory encoder. Training is single-timestep to mirror
inference, selecting one timestep per sequence (default: tail-biased toward
late timesteps). Labels come from the final outcome (terminal reward sign).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.nn import functional as F
from torch.optim import AdamW
from tqdm import tqdm

import wandb

import amago

from metamon.rl.pretrained import get_pretrained_model
from metamon.rl.train import create_offline_dataset, WANDB_ENTITY, WANDB_PROJECT
from metamon.interface import get_reward_function
from metamon.rl.timestep_win_probability_head import MetamonTstepWinProbAgent


@dataclass
class TrainConfig:
    model_name: str = "Abra"
    checkpoint: Optional[int] = None
    save_dir: str = "./winprob_runs"
    run_name: str = "tstep_winprob_abra"
    # data
    parsed_replay_dir: Optional[str] = None
    custom_replay_dir: Optional[str] = None
    custom_replay_sample_weight: float = 0.25
    formats: Optional[list[str]] = None
    reward_name: str = "BinaryReward"
    # optimization
    epochs: int = 5
    steps_per_epoch: int = 10000
    batch_size: int = 12
    grad_accum: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_steps: int = 2000
    mixed_precision: bool = True
    device: Optional[str] = None
    log: bool = True
    # timestep selection
    step_sampling: str = "tail_biased"  # one of: last, uniform, tail_biased, fixed_offset
    tail_gamma: float = 2.0  # >1 biases toward late steps
    fixed_offset_k: int = 0   # last_done - k (clamped >= 0)
    mix_uniform_prob: float = 0.0  # probability to use uniform instead of step_sampling
    calibrate_temperature: bool = False  # post-hoc temperature scaling on validation set


def last_true_idx(dones_bool_2d: torch.Tensor) -> torch.Tensor:
    """Robust last-True index per row for [B, L-1] dones.

    Falls back to the final transition if there are no True entries.
    """
    rev = torch.flip(dones_bool_2d, dims=[1]).int()
    idx_from_end = rev.argmax(dim=1)
    has_any = dones_bool_2d.any(dim=1)
    last = (dones_bool_2d.size(1) - 1) - idx_from_end
    fallback = torch.full_like(last, dones_bool_2d.size(1) - 1)
    return torch.where(has_any, last, fallback)


def build_label_from_batch(batch: amago.loading.Batch) -> torch.Tensor:
    """Outcome label per sequence from terminal reward.

    1 if final_reward>0, 0 if final_reward<0, 0.5 if final_reward==0 (tie/forfeit).
    """
    rews = batch.rews  # [B, L-1, 1]
    dones = batch.dones.squeeze(-1)  # [B, L-1]
    B, Lm1, _ = rews.shape
    last_idx = last_true_idx(dones.bool())  # [B]
    final_reward = rews[torch.arange(B, device=rews.device), last_idx, 0]  # [B]
    y = torch.where(
        final_reward > 0,
        torch.tensor(1.0, device=final_reward.device),
        torch.where(final_reward < 0, torch.tensor(0.0, device=final_reward.device), torch.tensor(0.5, device=final_reward.device)),
    )
    return y.unsqueeze(-1)


def _select_t_idxs(
    batch: amago.loading.Batch,
    *,
    mode: str = "tail_biased",
    tail_gamma: float = 2.0,
    fixed_offset_k: int = 0,
) -> torch.Tensor:
    """Select a single timestep per sequence.

    Returns:
        LongTensor [B] with indices in [0, last_done]. Always valid; no future.
    """
    device = batch.rl2s.device
    L = batch.rl2s.shape[1]
    last_done = last_true_idx(batch.dones.squeeze(-1).bool())  # [B]
    B = last_done.shape[0]

    if mode == "last":
        return last_done

    if mode == "fixed_offset":
        return torch.clamp(last_done - int(fixed_offset_k), min=0)

    # sample u ~ U(0,1); build per-item max (last_done + 1)
    # shape bookkeeping done in torch; avoid degenerate division.
    if mode == "uniform":
        # t ~ Uniform{0..last_done}
        # implement by sampling s in [0,1) then scaling by (last_done+1)
        u = torch.rand(B, device=device)
        t = torch.floor(u * (last_done + 1).float()).to(torch.long)
        return t

    if mode == "tail_biased":
        # bias towards later steps: s = u**(1/gamma); gamma>1 => late-heavy
        gamma = max(1e-6, float(tail_gamma))
        u = torch.rand(B, device=device)
        s = torch.pow(u, 1.0 / gamma)
        t = torch.floor(s * (last_done + 1).float()).to(torch.long)
        return t

    raise ValueError(f"Unknown step_sampling mode: {mode}")


def main(cfg: TrainConfig):
    os.makedirs(os.path.join(cfg.save_dir, cfg.run_name, "ckpts"), exist_ok=True)

    # 1) Load pretrained policy (frozen backbone)
    pretrained = get_pretrained_model(cfg.model_name)
    exp = pretrained.initialize_agent(checkpoint=cfg.checkpoint, log=False)
    policy = exp.policy  # unwrapped AMAGO Agent
    device = cfg.device or exp.DEVICE

    # 2) Attach timestep head; freeze base policy
    head_wrapper = MetamonTstepWinProbAgent(base_agent=policy)
    for p in policy.parameters():
        p.requires_grad = False
    policy.eval()

    optim = AdamW(head_wrapper.parameters_for_optimization, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 3) Dataset for labels (final outcome)
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

    @torch.no_grad()
    def validate(collect_logits: bool = False) -> dict:
        head_wrapper.head.eval()
        total = 0
        ys, ps = [], []
        z_list = []
        with torch.cuda.amp.autocast(enabled=cfg.mixed_precision and torch.cuda.is_available()):
            pbar_v = tqdm(total=cfg.val_steps, desc="Val", leave=False)
            while total < cfg.val_steps:
                samples = [amago_dset.sample_random_trajectory() for _ in range(cfg.batch_size)]
                batch = amago.loading.RLData_pad_collate(samples).to(device)
                # Evaluate at last pre-terminal timestep per sequence
                last_done = last_true_idx(batch.dones.squeeze(-1).bool())
                # current_step_logits_from_batch expects t_idxs; use last_done here
                _, logits_sel = head_wrapper.current_step_logits_from_batch(batch, last_done)
                p = torch.sigmoid(logits_sel).squeeze(-1)  # [B]
                y = build_label_from_batch(batch).squeeze(-1)  # [B]
                ys.append(y.detach().cpu())
                ps.append(p.detach().cpu())
                if collect_logits:
                    z_list.append(logits_sel.detach().cpu())
                total += y.shape[0]
                pbar_v.update(y.shape[0])
            pbar_v.close()

        y_all = torch.cat(ys, dim=0).numpy()
        p_all = torch.cat(ps, dim=0).numpy()
        z_all = torch.cat(z_list, dim=0).numpy() if collect_logits and z_list else None

        # Brier
        brier = float(((p_all - y_all) ** 2).mean())
        # Acc@0.5
        acc = float(((p_all >= 0.5) == (y_all >= 0.5)).mean())
        # AUC via rank statistic
        try:
            import numpy as np

            order = np.argsort(p_all)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(p_all)) + 1
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

        out = {"val/brier": brier, "val/acc@0.5": acc, "val/auc": auc}
        if z_all is not None:
            out["val/logits"] = z_all
            out["val/labels"] = y_all
        return out

    def fit_temperature(z: torch.Tensor, y: torch.Tensor) -> float:
        """Fit scalar temperature T>0 on logits z to minimize NLL on labels y."""
        T = torch.nn.Parameter(torch.ones(1, device=z.device))
        opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50)

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(z / T, y)
            loss.backward()
            return loss

        opt.step(closure)
        return float(T.detach().clamp_min(1e-6).cpu())

    best_auc = -1.0
    best_brier = float("inf")
    for epoch in range(cfg.epochs):
        head_wrapper.head.train()
        running_loss = 0.0
        pbar = tqdm(range(cfg.steps_per_epoch), desc=f"Epoch {epoch+1}/{cfg.epochs}")
        optim.zero_grad(set_to_none=True)
        for step_in_epoch in pbar:
            samples = [amago_dset.sample_random_trajectory() for _ in range(cfg.batch_size)]
            batch = amago.loading.RLData_pad_collate(samples).to(device)

            with torch.cuda.amp.autocast(enabled=cfg.mixed_precision and torch.cuda.is_available()):
                # select current step per sequence (no future, optionally tail-biased)
                choose_uniform = (cfg.mix_uniform_prob > 0.0) and (torch.rand(()) < cfg.mix_uniform_prob)
                mode = "uniform" if choose_uniform else cfg.step_sampling
                t_idxs = _select_t_idxs(
                    batch,
                    mode=mode,
                    tail_gamma=cfg.tail_gamma,
                    fixed_offset_k=cfg.fixed_offset_k,
                )
                _, logits_sel = head_wrapper.current_step_logits_from_batch(batch, t_idxs)
                # Progress-aware label smoothing toward 0.5 for early steps
                last_done = last_true_idx(batch.dones.squeeze(-1).bool())
                progress = (t_idxs.float() + 1) / (last_done.float() + 1)
                alpha_max = 0.10
                alpha = (1.0 - progress).clamp(0, 1) * alpha_max
                y = build_label_from_batch(batch)
                y_smooth = (1 - alpha).unsqueeze(-1) * y + alpha.unsqueeze(-1) * 0.5
                loss = F.binary_cross_entropy_with_logits(logits_sel, y_smooth)

            # backward
            scaler.scale(loss / cfg.grad_accum).backward()

            running_loss += float(loss.detach().item())
            if (step_in_epoch + 1) % cfg.grad_accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            pbar.set_postfix({"loss": running_loss / (step_in_epoch + 1)})

        # save head checkpoint (head-only)
        ckpt_path = os.path.join(cfg.save_dir, cfg.run_name, "ckpts", f"tstep_winprob_head_epoch_{epoch+1}.pt")
        torch.save(
            {
                "head_state_dict": head_wrapper.head.state_dict(),
                "base_model": cfg.model_name,
                "checkpoint": cfg.checkpoint or pretrained.default_checkpoint,
                "config": cfg.__dict__,
            },
            ckpt_path,
        )
        print(f"[INFO] Saved {ckpt_path}")

        metrics = validate(collect_logits=cfg.calibrate_temperature)
        print(
            f"[VAL] epoch={epoch+1} auc={metrics['val/auc']:.4f} brier={metrics['val/brier']:.4f} acc@0.5={metrics['val/acc@0.5']:.4f}"
        )
        if cfg.log:
            wandb.log({"train/loss": running_loss / max(1, cfg.steps_per_epoch), **metrics, "epoch": epoch + 1})

        # track best by AUC (tie-breaker Brier)
        auc = metrics["val/auc"]
        brier = metrics["val/brier"]
        is_best = False
        if auc == auc:  # not NaN
            if auc > best_auc or (abs(auc - best_auc) < 1e-6 and brier < best_brier):
                best_auc, best_brier, is_best = auc, brier, True
        elif brier < best_brier:
            best_brier, is_best = brier, True

        if is_best:
            # Optional temperature scaling using collected validation logits/labels
            temperature = 1.0
            if cfg.calibrate_temperature and "val/logits" in metrics:
                import numpy as np
                z_np = metrics["val/logits"]
                y_np = metrics["val/labels"]
                z = torch.from_numpy(z_np).to(device).float()
                y = torch.from_numpy(y_np).to(device).float().unsqueeze(-1)
                temperature = fit_temperature(z, y)
            best_path = os.path.join(cfg.save_dir, cfg.run_name, "ckpts", "tstep_winprob_head_best.pt")
            torch.save(
                {
                    "head_state_dict": head_wrapper.head.state_dict(),
                    "base_model": cfg.model_name,
                    "checkpoint": cfg.checkpoint or pretrained.default_checkpoint,
                    "config": cfg.__dict__,
                    "metrics": {k: v for k, v in metrics.items() if not k.startswith("val/logits") and not k.startswith("val/labels")},
                    "temperature": temperature,
                },
                best_path,
            )
            print(f"[INFO] Updated best checkpoint: {best_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a timestep-only WinProbability head on top of a frozen pretrained model"
    )
    parser.add_argument(
        "--model",
        default="Abra",
        choices=["Abra", "SmallRL", "MediumRL", "SyntheticRLV2", "Minikazam"],
        help="Pretrained model name",
    )
    parser.add_argument("--ckpt", type=int, default=None, help="Checkpoint number to load (defaults to the model default)")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps_per_epoch", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--reward_name",
        type=str,
        default="BinaryReward",
        choices=["BinaryReward", "AggressiveShapedReward", "DefaultShapedReward"],
    )
    parser.add_argument("--parsed_replay_dir", type=str, default=None)
    parser.add_argument("--custom_replay_dir", type=str, default=None)
    parser.add_argument("--custom_replay_sample_weight", type=float, default=0.25)
    parser.add_argument("--formats", type=str, nargs="+", default=None)
    parser.add_argument("--no_amp", action="store_true")
    # timestep selection
    parser.add_argument(
        "--step_sampling",
        type=str,
        default="tail_biased",
        choices=["last", "uniform", "tail_biased", "fixed_offset"],
        help="How to select the single timestep per sequence for training",
    )
    parser.add_argument("--tail_gamma", type=float, default=2.0, help="Gamma>1 biases tail_biased sampling toward late steps")
    parser.add_argument("--fixed_offset_k", type=int, default=0, help="For fixed_offset: use last_done - k (clamped >= 0)")
    parser.add_argument("--mix_uniform_prob", type=float, default=0.0, help="Probability to sample uniform step instead of --step_sampling")
    parser.add_argument("--calibrate_temperature", action="store_true", help="Fit a scalar temperature on validation logits and save it")
    args = parser.parse_args()

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
        step_sampling=args.step_sampling,
        tail_gamma=args.tail_gamma,
        fixed_offset_k=args.fixed_offset_k,
        mix_uniform_prob=args.mix_uniform_prob,
        calibrate_temperature=args.calibrate_temperature,
    )
    main(cfg)
