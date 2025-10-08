#!/usr/bin/env bash
set -Eeuo pipefail

# Ensure logs are flushed promptly in Cloud Run Jobs
export PYTHONUNBUFFERED=1

# Self-play battle runner with environment variable support
# Can be overridden by passing CLI arguments directly
#
# HuggingFace Authentication:
#   Set HF_TOKEN environment variable to avoid rate limits when downloading models.
#   The huggingface_hub library automatically respects this variable.
#   Example: export HF_TOKEN=hf_xxxxxxxxxxxxx
#   Or in Cloud Run: --set-env-vars HF_TOKEN=hf_xxxxxxxxxxxxx

: "${AGENT1:=SyntheticRLV1}"
: "${AGENT2:=SyntheticRLV2}"
: "${BATTLE_FORMAT:=gen1ou}"
: "${NUM_BATTLES:=100}"
: "${TEAM_SET:=competitive}"
: "${CONCURRENCY:=1}"
: "${USERNAME1:=SelfPlayAgent1}"
: "${USERNAME2:=SelfPlayAgent2}"
: "${BATTLE_BACKEND:=poke-env}"
: "${DELAY_AGENT2:=5}"

# Optional environment variables (no defaults)
SAVE_TRAJECTORIES_TO="${SAVE_TRAJECTORIES_TO:-}"
AGENT1_LOCAL_PATH="${AGENT1_LOCAL_PATH:-}"
AGENT1_CHECKPOINT="${AGENT1_CHECKPOINT:-}"
AGENT1_MODEL_GIN="${AGENT1_MODEL_GIN:-}"
AGENT1_TRAIN_GIN="${AGENT1_TRAIN_GIN:-}"
AGENT2_LOCAL_PATH="${AGENT2_LOCAL_PATH:-}"
AGENT2_CHECKPOINT="${AGENT2_CHECKPOINT:-}"
AGENT2_MODEL_GIN="${AGENT2_MODEL_GIN:-}"
AGENT2_TRAIN_GIN="${AGENT2_TRAIN_GIN:-}"
SEQUENTIAL="${SEQUENTIAL:-}"

# Build command arguments from environment variables
CMD_ARGS=()

# If CLI args provided, use them directly and skip env var parsing
if [[ $# -gt 0 ]]; then
  exec python examples/self_play_battle.py "$@"
fi

# Build args from env vars
[[ -n "$AGENT1" ]] && CMD_ARGS+=(--agent1 "$AGENT1")
[[ -n "$AGENT2" ]] && CMD_ARGS+=(--agent2 "$AGENT2")
[[ -n "$BATTLE_FORMAT" ]] && CMD_ARGS+=(--battle_format "$BATTLE_FORMAT")
[[ -n "$NUM_BATTLES" ]] && CMD_ARGS+=(--num_battles "$NUM_BATTLES")
[[ -n "$TEAM_SET" ]] && CMD_ARGS+=(--team_set "$TEAM_SET")
[[ -n "$USERNAME1" ]] && CMD_ARGS+=(--username1 "$USERNAME1")
[[ -n "$USERNAME2" ]] && CMD_ARGS+=(--username2 "$USERNAME2")
[[ -n "$BATTLE_BACKEND" ]] && CMD_ARGS+=(--battle_backend "$BATTLE_BACKEND")
[[ -n "$DELAY_AGENT2" ]] && CMD_ARGS+=(--delay_agent2 "$DELAY_AGENT2")

# Optional arguments
[[ -n "$SAVE_TRAJECTORIES_TO" ]] && CMD_ARGS+=(--save_trajectories_to "$SAVE_TRAJECTORIES_TO")
[[ -n "$AGENT1_LOCAL_PATH" ]] && CMD_ARGS+=(--agent1_local_path "$AGENT1_LOCAL_PATH")
[[ -n "$AGENT1_CHECKPOINT" ]] && CMD_ARGS+=(--agent1_checkpoint "$AGENT1_CHECKPOINT")
[[ -n "$AGENT1_MODEL_GIN" ]] && CMD_ARGS+=(--agent1_model_gin "$AGENT1_MODEL_GIN")
[[ -n "$AGENT1_TRAIN_GIN" ]] && CMD_ARGS+=(--agent1_train_gin "$AGENT1_TRAIN_GIN")
[[ -n "$AGENT2_LOCAL_PATH" ]] && CMD_ARGS+=(--agent2_local_path "$AGENT2_LOCAL_PATH")
[[ -n "$AGENT2_CHECKPOINT" ]] && CMD_ARGS+=(--agent2_checkpoint "$AGENT2_CHECKPOINT")
[[ -n "$AGENT2_MODEL_GIN" ]] && CMD_ARGS+=(--agent2_model_gin "$AGENT2_MODEL_GIN")
[[ -n "$AGENT2_TRAIN_GIN" ]] && CMD_ARGS+=(--agent2_train_gin "$AGENT2_TRAIN_GIN")
[[ -n "$SEQUENTIAL" ]] && CMD_ARGS+=(--sequential)

echo "Starting self-play battles with args: ${CMD_ARGS[*]}"
echo "Resolved Showdown endpoint: ${SHOWDOWN_HOST:-127.0.0.1}:${SHOWDOWN_PORT:-${PORT:-8000}}"
exec python -u examples/self_play_battle.py "${CMD_ARGS[@]}"
