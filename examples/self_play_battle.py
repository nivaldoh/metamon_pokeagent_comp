#!/usr/bin/env python3
"""
Self-play battle script for Metamon.

Launches two agents (potentially different models) on the local Showdown ladder
to battle against each other. Supports both pretrained HuggingFace models and
locally trained models.

Prerequisites:
- Pokemon Showdown server running locally:
  cd server/pokemon-showdown && node pokemon-showdown start --no-security &
- Environment variable set: export METAMON_CACHE_DIR=cache

Example Usage:
    # 1. Self-play: Same HF model vs itself (generate training data)
    python examples/self_play_battle.py \
        --agent1 SyntheticRLV1 \
        --agent2 SyntheticRLV2 \
        --battle_format gen1ou \
        --num_battles 10000 \
        --save_trajectories_to ./self_play/synrl_selfplay_data

    # 2. Two different HF models
    python examples/self_play_battle.py \
        --agent1 SyntheticRLV2 \
        --agent2 MediumRL \
        --battle_format gen1ou \
        --num_battles 50

    # 3. Local model vs HF model

    # 4. Local model vs local model
    python examples/self_play_battle.py \
        --agent1_local_path saved_checkpoints/abra-semantic2-rl \
        --agent1_model_gin medium_multitask_semantic.gin \
        --agent1_train_gin binary_rl.gin \
        --agent1_checkpoint 10 \
        --agent2_local_path saved_checkpoints/abra-semantic2-rl \
        --agent2_model_gin medium_multitask_semantic.gin \
        --agent2_train_gin binary_rl.gin \
        --agent2_checkpoint 14 \
        --battle_format gen9ou \
        --num_battles 5 \
        --save_trajectories_to ./local_vs_local_data


    # 5. Custom usernames and team sets
    python examples/self_play_battle.py \
        --agent1 SmallRL \
        --agent2 MediumRL \
        --username1 MyBot1 \
        --username2 MyBot2 \
        --team_set modern_replays \
        --battle_format gen3ou \
        --num_battles 10
"""

import os
import sys
import multiprocessing as mp
from argparse import ArgumentParser
from typing import Optional, Dict, Any

import metamon
from metamon.rl.pretrained import get_pretrained_model_names
from metamon.rl.evaluate import (
    pretrained_vs_pretrained_headtohead,
)
 


 


def _agent_spec_from_args(args, agent_num: int) -> Dict[str, Any]:
    if agent_num == 1:
        return {
            "hf_name": args.agent1,
            "local_path": args.agent1_local_path,
            "checkpoint": args.agent1_checkpoint,
            "model_gin": args.agent1_model_gin,
            "train_gin": args.agent1_train_gin,
        }
    else:
        return {
            "hf_name": args.agent2,
            "local_path": args.agent2_local_path,
            "checkpoint": args.agent2_checkpoint,
            "model_gin": args.agent2_model_gin,
            "train_gin": args.agent2_train_gin,
        }


def main():
    parser = ArgumentParser(
        description="Run self-play battles between two Metamon agents on the local ladder."
    )

    # Agent 1 configuration
    parser.add_argument(
        "--agent1",
        type=str,
        default=None,
        choices=get_pretrained_model_names() + [None],
        help="Name of pretrained HuggingFace model for Agent 1",
    )
    parser.add_argument(
        "--agent1_local_path",
        type=str,
        default=None,
        help="Path to local checkpoint directory for Agent 1 (alternative to --agent1)",
    )
    parser.add_argument(
        "--agent1_checkpoint",
        type=int,
        default=None,
        help="Checkpoint number to use for Agent 1",
    )
    parser.add_argument(
        "--agent1_model_gin",
        type=str,
        default=None,
        help="Model gin config for local Agent 1 (required if using --agent1_local_path)",
    )
    parser.add_argument(
        "--agent1_train_gin",
        type=str,
        default=None,
        help="Training gin config for local Agent 1 (required if using --agent1_local_path)",
    )

    # Agent 2 configuration
    parser.add_argument(
        "--agent2",
        type=str,
        default=None,
        choices=get_pretrained_model_names() + [None],
        help="Name of pretrained HuggingFace model for Agent 2",
    )
    parser.add_argument(
        "--agent2_local_path",
        type=str,
        default=None,
        help="Path to local checkpoint directory for Agent 2 (alternative to --agent2)",
    )
    parser.add_argument(
        "--agent2_checkpoint",
        type=int,
        default=None,
        help="Checkpoint number to use for Agent 2",
    )
    parser.add_argument(
        "--agent2_model_gin",
        type=str,
        default=None,
        help="Model gin config for local Agent 2 (required if using --agent2_local_path)",
    )
    parser.add_argument(
        "--agent2_train_gin",
        type=str,
        default=None,
        help="Training gin config for local Agent 2 (required if using --agent2_local_path)",
    )

    # Battle configuration
    parser.add_argument(
        "--battle_format",
        type=str,
        default="gen1ou",
        help="Battle format (e.g., gen1ou, gen2ou, gen9ou)",
    )
    parser.add_argument(
        "--team_set",
        type=str,
        default="competitive",
        help="Team set to use. Options: " + ", ".join(metamon.env.METAMON_TEAM_SETS),
    )
    parser.add_argument(
        "--num_battles",
        type=int,
        default=10,
        help="Number of battles for each agent to play",
    )
    parser.add_argument(
        "--battle_backend",
        type=str,
        default="poke-env",
        choices=["poke-env", "metamon"],
        help="Battle backend to use",
    )
    parser.add_argument(
        "--save_trajectories_to",
        type=str,
        default=None,
        help="Directory to save battle trajectories (in parsed replay format)",
    )

    # Agent usernames
    parser.add_argument(
        "--username1",
        type=str,
        default="SelfPlayAgent1",
        help="Username for Agent 1 on the ladder",
    )
    parser.add_argument(
        "--username2",
        type=str,
        default="SelfPlayAgent2",
        help="Username for Agent 2 on the ladder",
    )

    # Execution options (keep a small delay to help pairing)
    parser.add_argument(
        "--delay_agent2",
        type=int,
        default=5,
        help="Seconds to delay before starting Agent 2 (helps ensure Agent 1 is ready)",
    )

    args = parser.parse_args()

    # Validation
    if args.agent1 is None and args.agent1_local_path is None:
        parser.error("Must specify either --agent1 or --agent1_local_path")
    if args.agent2 is None and args.agent2_local_path is None:
        parser.error("Must specify either --agent2 or --agent2_local_path")
    if args.username1 == args.username2:
        parser.error("Agent usernames must be different! Use --username1 and --username2")

    # Check cache dir is set
    if metamon.METAMON_CACHE_DIR is None:
        print("Error: Please set METAMON_CACHE_DIR environment variable")
        print("Example: export METAMON_CACHE_DIR=/tmp/metamon_cache")
        sys.exit(1)

    print("=" * 80)
    print("METAMON SELF-PLAY BATTLE")
    print("=" * 80)

    # Create trajectory save directory if needed
    if args.save_trajectories_to:
        os.makedirs(args.save_trajectories_to, exist_ok=True)
        print(f"\nTrajectories will be saved to: {args.save_trajectories_to}")

    print("\n" + "=" * 80)
    print("STARTING SELF-PLAY BATTLES")
    print("=" * 80)
    print(f"Agent 1: {args.agent1 or args.agent1_local_path} (username: {args.username1})")
    print(f"Agent 2: {args.agent2 or args.agent2_local_path} (username: {args.username2})")
    print(f"Battle format: {args.battle_format}")
    print(f"Battles per agent: {args.num_battles}")
    print("=" * 80 + "\n")

    # Always run exactly two agents simultaneously via evaluation helper
    print("Running two agents head-to-head (direct orchestration)")
    print(f"Agent 2 will start {args.delay_agent2} seconds after Agent 1\n")

    spec1 = _agent_spec_from_args(args, 1)
    spec2 = _agent_spec_from_args(args, 2)
    results = pretrained_vs_pretrained_headtohead(
        agent1_spec=spec1,
        agent2_spec=spec2,
        username1=args.username1,
        username2=args.username2,
        battle_format=args.battle_format,
        team_set_name=args.team_set,
        total_battles=args.num_battles,
        battle_backend=args.battle_backend,
        save_trajectories_to=args.save_trajectories_to,
        delay_agent2=args.delay_agent2,
    )

    print("\n" + "=" * 80)
    print("SELF-PLAY BATTLES COMPLETED")
    print("=" * 80)

    if args.save_trajectories_to:
        print(f"\nTrajectories saved to:")
        print(f"  Agent 1: {args.save_trajectories_to}/agent1/{args.battle_format}/")
        print(f"  Agent 2: {args.save_trajectories_to}/agent2/{args.battle_format}/")
        print("\nYou can use this data for training with:")
        print(f"  python -m metamon.rl.train --custom_replay_dir {args.save_trajectories_to}/agent1")
        print(f"  # or mix both agents:")
        print(f"  # Combine directories first, then use --custom_replay_dir")

    print("\nDone!")
    return results


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()
