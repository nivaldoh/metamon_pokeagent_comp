#!/usr/bin/env python3
"""
Self-play battle script for Metamon.

Launches two agents (potentially different models) on the local Showdown ladder
to battle against each other. Supports both pretrained HuggingFace models and
locally trained models.

Prerequisites:
- Pokemon Showdown server running locally:
  cd server/pokemon-showdown && node pokemon-showdown start --no-security &
- Environment variable set: export METAMON_CACHE_DIR=/path/to/storage

Example Usage:
    # 1. Self-play: Same HF model vs itself (generate training data)
    python examples/self_play_battle.py \
        --agent1 SyntheticRLV2 \
        --agent2 SyntheticRLV2 \
        --battle_format gen1ou \
        --num_battles 100 \
        --save_trajectories_to ./selfplay_data

    # 2. Two different HF models
    python examples/self_play_battle.py \
        --agent1 SmallRL \
        --agent2 MediumRL \
        --battle_format gen2ou \
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
        --agent2_checkpoint 15 \
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
import time
import multiprocessing as mp
from argparse import ArgumentParser
from typing import Optional, Dict, Any

import metamon
from metamon.rl.pretrained import get_pretrained_model, get_pretrained_model_names
from metamon.rl.evaluate import pretrained_vs_local_ladder
from metamon.env import get_metamon_teams


def load_model_from_args(args, agent_num: int):
    """
    Load a model based on command line arguments.

    Args:
        args: Parsed command line arguments
        agent_num: 1 or 2, indicating which agent to load

    Returns:
        PretrainedModel instance (HF or local)
    """
    if agent_num == 1:
        agent_name = args.agent1
        local_path = args.agent1_local_path
        checkpoint = args.agent1_checkpoint
        model_gin = args.agent1_model_gin
        train_gin = args.agent1_train_gin
    else:
        agent_name = args.agent2
        local_path = args.agent2_local_path
        checkpoint = args.agent2_checkpoint
        model_gin = args.agent2_model_gin
        train_gin = args.agent2_train_gin

    # Load from HuggingFace
    if agent_name is not None:
        print(f"Loading Agent {agent_num} from HuggingFace: {agent_name}")
        return get_pretrained_model(agent_name), checkpoint

    # Load from local path
    elif local_path is not None:
        from metamon.rl.pretrained import LocalPretrainedModel

        print(f"Loading Agent {agent_num} from local path: {local_path}")

        if model_gin is None or train_gin is None:
            raise ValueError(
                f"When using local models, must specify both --agent{agent_num}_model_gin "
                f"and --agent{agent_num}_train_gin"
            )

        # Determine model name from path
        model_name = os.path.basename(local_path.rstrip('/'))

        # Validate/find checkpoint file(s) for local runs
        ckpt_dir = os.path.join(os.path.dirname(local_path), model_name, "ckpts", "policy_weights")
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(
                f"Expected checkpoint directory not found for Agent {agent_num}: {ckpt_dir}"
            )

        # Enumerate available checkpoint epochs
        available_epochs = []
        try:
            for fn in os.listdir(ckpt_dir):
                if fn.startswith("policy_epoch_") and fn.endswith(".pt"):
                    try:
                        num = int(fn[len("policy_epoch_") : -len(".pt")])
                        available_epochs.append(num)
                    except ValueError:
                        continue
        except Exception:
            pass

        if not available_epochs:
            raise FileNotFoundError(
                f"No checkpoint files found for Agent {agent_num} in {ckpt_dir}"
            )

        # Choose checkpoint: respect requested if present, else fall back to latest
        if checkpoint is None:
            checkpoint = max(available_epochs)
            print(
                f"[Agent {agent_num}] No checkpoint specified; using latest available: epoch {checkpoint}"
            )
        elif checkpoint not in available_epochs:
            fallback = max([e for e in available_epochs if e <= checkpoint] or available_epochs)
            print(
                f"[Agent {agent_num}] Requested checkpoint epoch {checkpoint} not found in {ckpt_dir}. "
                f"Falling back to epoch {fallback} (available: {sorted(available_epochs)})."
            )
            checkpoint = fallback

        # Create LocalPretrainedModel
        # Note: This requires knowing the obs/action spaces used during training
        # We'll use defaults, but users may need to customize this
        model = LocalPretrainedModel(
            amago_ckpt_dir=os.path.dirname(local_path),
            model_name=model_name,
            model_gin_config=model_gin,
            train_gin_config=train_gin,
            default_checkpoint=checkpoint,
            # These should match what was used during training
            # Users can modify the script if they used custom spaces
            action_space=metamon.interface.DefaultActionSpace(),
            observation_space=metamon.interface.ExpandedObservationSpace(),
            tokenizer=metamon.tokenizer.get_tokenizer("DefaultObservationSpace-v1"),
        )
        return model, checkpoint

    else:
        raise ValueError(f"Must specify either --agent{agent_num} or --agent{agent_num}_local_path")


def run_agent_on_ladder(
    model,
    checkpoint: Optional[int],
    username: str,
    battle_format: str,
    team_set,
    num_battles: int,
    battle_backend: str,
    save_trajectories_to: Optional[str],
    agent_id: int,
) -> Dict[str, Any]:
    """
    Run a single agent on the local ladder.

    This function is designed to run in a separate process.
    """
    print(f"\n[Agent {agent_id}] Starting with username: {username}")
    print(f"[Agent {agent_id}] Battle format: {battle_format}")
    print(f"[Agent {agent_id}] Number of battles: {num_battles}")

    # Set up trajectory saving path for this agent
    if save_trajectories_to:
        agent_traj_path = os.path.join(save_trajectories_to, f"agent{agent_id}")
        os.makedirs(agent_traj_path, exist_ok=True)
        print(f"[Agent {agent_id}] Saving trajectories to: {agent_traj_path}")
    else:
        agent_traj_path = None

    try:
        # Run the agent on the ladder
        results = pretrained_vs_local_ladder(
            pretrained_model=model,
            username=username,
            battle_format=battle_format,
            team_set=team_set,
            total_battles=num_battles,
            checkpoint=checkpoint,
            battle_backend=battle_backend,
            save_trajectories_to=agent_traj_path,
            log_to_wandb=False,
        )

        print(f"\n[Agent {agent_id}] Completed battles!")
        print(f"[Agent {agent_id}] Results: {results}")

        return results

    except Exception as e:
        print(f"\n[Agent {agent_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def run_agent_worker(agent_num: int, args) -> Dict[str, Any]:
    """
    Top-level worker entrypoint for multiprocessing.

    Reconstructs the model and team set inside the child process to avoid
    pickling issues from the 'spawn' start method.
    """
    # Optional stagger for agent 2 to ensure both are online together.
    if getattr(args, "delay_agent2", 0) and agent_num == 2:
        time.sleep(args.delay_agent2)

    # Build model/checkpoint inside the worker
    model, checkpoint = load_model_from_args(args, agent_num=agent_num)

    # Build team set inside the worker
    team_set = get_metamon_teams(args.battle_format, args.team_set)

    username = args.username1 if agent_num == 1 else args.username2

    return run_agent_on_ladder(
        model=model,
        checkpoint=checkpoint,
        username=username,
        battle_format=args.battle_format,
        team_set=team_set,
        num_battles=args.num_battles,
        battle_backend=args.battle_backend,
        save_trajectories_to=args.save_trajectories_to,
        agent_id=agent_num,
    )


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

    # Execution options
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run agents sequentially instead of in parallel (useful for debugging)",
    )
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

    # Load models
    model1, checkpoint1 = load_model_from_args(args, agent_num=1)
    model2, checkpoint2 = load_model_from_args(args, agent_num=2)

    # Load team set
    print(f"\nLoading team set: {args.team_set}")
    team_set = get_metamon_teams(args.battle_format, args.team_set)

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

    if args.sequential:
        # Sequential execution (mainly for debugging)
        print("Running in SEQUENTIAL mode (Agent 1 first, then Agent 2)")
        print("\nNote: Sequential mode won't generate self-play data!")
        print("The agents won't be online at the same time.\n")

        result1 = run_agent_on_ladder(
            model1, checkpoint1, args.username1, args.battle_format,
            team_set, args.num_battles, args.battle_backend,
            args.save_trajectories_to, agent_id=1
        )

        result2 = run_agent_on_ladder(
            model2, checkpoint2, args.username2, args.battle_format,
            team_set, args.num_battles, args.battle_backend,
            args.save_trajectories_to, agent_id=2
        )

        results = {"agent1": result1, "agent2": result2}

    else:
        # Parallel execution using multiprocessing
        print("Running in PARALLEL mode (both agents simultaneously)")
        print(f"Agent 2 will start {args.delay_agent2} seconds after Agent 1\n")

        # Use spawn context for clean process separation
        ctx = mp.get_context('spawn')

        # Start both agents with top-level worker targets (pickleable)
        process1 = ctx.Process(target=run_agent_worker, args=(1, args))
        process2 = ctx.Process(target=run_agent_worker, args=(2, args))

        process1.start()
        process2.start()

        # Wait for both to complete
        print("\nWaiting for agents to complete battles...")
        process1.join()
        process2.join()

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

        results = {
            "status": "completed",
            "note": "Results are printed above by each agent process"
        }

    print("\nDone!")
    return results


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()
