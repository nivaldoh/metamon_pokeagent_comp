# Metamon Docker & Cloud Run Deployment

This directory contains Docker configuration and deployment scripts for running Metamon self-play battles in containerized environments, especially Google Cloud Run.

## Files

- **`Dockerfile`**: Multi-stage Docker image with Python, Node.js, Pokemon Showdown server, and Metamon
- **`with_showdown.sh`**: Wrapper script that starts Pokemon Showdown and waits for it to be ready
- **`run_self_play.sh`**: Entry point for self-play battles with environment variable support
- **`deploy_cloud_run.sh`**: Example deployment script for Google Cloud Run
- **`.dockerignore`**: Files excluded from Docker build context

## Quick Start

### 1. Build the Docker Image

```bash
# Using Cloud Build (recommended for GCP)
gcloud builds submit --config cloudbuild.yaml

# Or build locally
docker build -t metamon-selfplay .
```

### 2. Run Locally

```bash
# Basic run with defaults (SyntheticRLV1 vs SyntheticRLV2, gen1ou, 100 battles)
docker run metamon-selfplay

# With custom parameters via environment variables
docker run -e AGENT1=SyntheticRLV2 -e NUM_BATTLES=500 metamon-selfplay

# With custom parameters via CLI arguments
docker run metamon-selfplay --agent1 MediumRL --battle_format gen2ou --num_battles 50

# With HuggingFace authentication to avoid rate limits
docker run -e HF_TOKEN=hf_xxxxxxxxxxxxx metamon-selfplay
```

### 3. Deploy to Cloud Run

```bash
# Configure the deployment script
export PROJECT_ID=your-gcp-project
export REGION=us-central1

# Deploy with default settings
./docker/deploy_cloud_run.sh
```

## HuggingFace Authentication

### Why Authenticate?

When downloading pretrained models from HuggingFace Hub, unauthenticated requests have lower rate limits:
- **Without token**: ~1,000 requests/hour per IP
- **With token**: ~1,000,000 requests/hour

For production deployments or high-concurrency runs, authentication is strongly recommended.

### How to Authenticate

The code automatically respects the `HF_TOKEN` environment variable. No code changes needed!

#### Get Your Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access is sufficient)
3. Copy the token (starts with `hf_`)

#### Local Development

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxx
docker run -e HF_TOKEN=$HF_TOKEN metamon-selfplay
```

#### Cloud Run - Option 1: Secret Manager (Recommended)

Store your token securely in Google Secret Manager:

```bash
# Create secret
echo -n "hf_xxxxxxxxxxxxx" | gcloud secrets create huggingface-token --data-file=-

# Deploy with secret
gcloud run deploy metamon-selfplay \
  --image your-image \
  --set-secrets HF_TOKEN=huggingface-token:latest \
  --args="--agent1=SyntheticRLV1,--agent2=SyntheticRLV2,--num_battles=1000"
```

#### Cloud Run - Option 2: Direct Environment Variable

Simpler but less secure (token visible in deployment config):

```bash
gcloud run deploy metamon-selfplay \
  --image your-image \
  --set-env-vars HF_TOKEN=hf_xxxxxxxxxxxxx \
  --args="--agent1=SyntheticRLV1,--agent2=SyntheticRLV2,--num_battles=1000"
```

## Configuration

### Environment Variables

All parameters from `examples/self_play_battle.py` can be set via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT1` | `SyntheticRLV1` | HuggingFace model name for agent 1 |
| `AGENT2` | `SyntheticRLV2` | HuggingFace model name for agent 2 |
| `BATTLE_FORMAT` | `gen1ou` | Battle format (gen1ou, gen2ou, etc.) |
| `NUM_BATTLES` | `100` | Number of battles per agent |
| `TEAM_SET_1` | `competitive` | Team set for agent 1 |
| `TEAM_SET_2` | `competitive` | Team set for agent 2 |
| `CONCURRENCY` | `1` | Number of parallel agent pairs |
| `USERNAME1` | `SelfPlayAgent1` | Username for agent 1 |
| `USERNAME2` | `SelfPlayAgent2` | Username for agent 2 |
| `BATTLE_BACKEND` | `poke-env` | Battle backend (poke-env or metamon) |
| `DELAY_AGENT2` | `5` | Delay before starting agent 2 (seconds) |
| `SAVE_TRAJECTORIES_TO` | *(none)* | Directory to save trajectories |
| `HF_TOKEN` | *(none)* | HuggingFace API token (optional) |
| `SEQUENTIAL` | *(none)* | Set to any value for sequential mode |

#### Local Models

For local checkpoints, use these additional variables:

| Variable | Description |
|----------|-------------|
| `AGENT1_LOCAL_PATH` | Path to local checkpoint for agent 1 |
| `AGENT1_CHECKPOINT` | Checkpoint epoch number |
| `AGENT1_MODEL_GIN` | Model gin config file |
| `AGENT1_TRAIN_GIN` | Training gin config file |
| *(same for AGENT2)* | |

### CLI Arguments

CLI arguments override environment variables and provide more flexibility:

```bash
docker run metamon-selfplay \
  --agent1 SyntheticRLV2 \
  --agent2 MediumRL \
  --battle_format gen3ou \
  --team_set_1 modern_replays \
  --team_set_2 paper_replays \
  --num_battles 1000 \
  --save_trajectories_to /cache/trajectories
```

See `python examples/self_play_battle.py --help` for all available arguments.

## Cloud Run Best Practices

### Resource Allocation

- **Small models (SyntheticRL, SmallRL)**: 2 vCPU, 4Gi memory
- **Medium models (MediumRL)**: 4 vCPU, 8Gi memory
- **Large models (LargeRL)**: 8 vCPU, 16Gi memory

### Timeout

Self-play battles can run for a long time. Set appropriate timeouts:

```bash
--timeout 3600  # 1 hour
--timeout 7200  # 2 hours
```

### Cost Optimization

- Use `--max-instances` to limit concurrent containers
- Set `--min-instances 0` for auto-scaling to zero
- Use `--no-allow-unauthenticated` and trigger via Cloud Scheduler/Tasks

## Troubleshooting

### Rate Limit Errors

**Error**: `429 Too Many Requests` when downloading models

**Solution**: Set `HF_TOKEN` environment variable (see authentication section above)

### Pokemon Showdown Fails to Start

**Error**: `Pokémon Showdown failed to become ready on port 8080`

**Solution**:
- Check that Node.js and npm versions are compatible
- Verify the Showdown build completed successfully during image build
- Increase memory allocation (Showdown needs ~1Gi minimum)

### Out of Memory

**Error**: Container killed with exit code 137

**Solution**:
- Increase Cloud Run memory allocation

### Models Download Too Slowly

**Issue**: First run takes a long time downloading models

**Solution**:
- Models are cached in `/cache` (mapped to `METAMON_CACHE_DIR`)
- Consider using a persistent disk or Cloud Storage FUSE mount for the cache
- Pre-warm the cache by running a short test battle first

## Advanced Usage

### Cloud Run Jobs: Manifest-Driven Tasks

Provide a JSON Lines manifest under the repository's `manifests/` directory and set `JSONL_TASK_MANIFEST` to the filename. The job task at index `CLOUD_RUN_TASK_INDEX` (0-based) reads the corresponding line and exports parameters to env vars.

Example manifest (`params.jsonl`):

```
{"battle_format":"gen1ou","agent1":"SyntheticRLV1","agent2":"SyntheticRLV2","team_set_1":"competitive","team_set_2":"competitive","num_battles":100,"cfg_tag":"gen1ou_SyntheticRLV1_competitive__vs__competitive_SyntheticRLV2"}
{"battle_format":"gen1ou","agent1":"SyntheticRLV1","agent2":"SyntheticRLV2","team_set_1":"modern_replays","team_set_2":"paper_replays","num_battles":100,"cfg_tag":"gen1ou_SyntheticRLV1_modern_replays__vs__paper_replays_SyntheticRLV2"}
{"battle_format":"gen9ou","agent1":"Abra","agent2":"Abra","team_set_1":"modern_replays","team_set_2":"modern_replays","num_battles":50,"cfg_tag":"gen9ou_Abra_modern_replays__vs__modern_replays_Abra"}
```

Supported keys map to env vars used by the runner:
- `agent1` → `AGENT1`
- `agent2` → `AGENT2`
- `battle_format` → `BATTLE_FORMAT`
- `team_set_1` → `TEAM_SET_1`
- `team_set_2` → `TEAM_SET_2`
- `num_battles` → `NUM_BATTLES`
- `cfg_tag` → `CFG_TAG` (optional; defaults to `<format>_<agent1>_<team_set_1>__vs__<team_set_2>_<agent2>`)

Output directory: when `JSONL_TASK_MANIFEST` is set, `run_self_play.sh` constructs a per-task save path automatically using a UTC run id and config tag. The base path is controlled by `OUTPUT_BASE` (defaults to `/selfplay`):

```
<OUTPUT_BASE>/run=<RUN_ID>/cfg=<CFG_TAG>/task-<CLOUD_RUN_TASK_INDEX>
```

Override `OUTPUT_BASE`, `RUN_ID`, or `CFG_TAG` via env vars if you want stable naming.

Minimal Cloud Run Job example:

```
gcloud run jobs create metamon-selfplay-job \
  --image ${IMAGE_URL} \
  --region ${REGION} \
  --tasks 3 \
  --set-env-vars JSONL_TASK_MANIFEST=selfplay_rlv2.jsonl \
  --set-secrets HF_TOKEN=huggingface-token:latest \
  --args=run-self-play

# Ensure your image includes `manifests/selfplay_rlv2.jsonl` at build time, or set `MANIFESTS_DIR` to point elsewhere in the container.
```

### Saving Trajectories to Cloud Storage

```bash
# Mount Cloud Storage bucket (requires GCS FUSE)
gcloud run deploy metamon-selfplay \
  --image your-image \
  --set-env-vars SAVE_TRAJECTORIES_TO=/mnt/gcs/trajectories \
  --execution-environment gen2
  # Then configure Cloud Storage FUSE mount
```

### Scheduled Batch Jobs

Use Cloud Scheduler to trigger periodic data generation:

```bash
gcloud scheduler jobs create http generate-daily-data \
  --schedule="0 2 * * *" \
  --uri="https://metamon-selfplay-xxx.run.app" \
  --http-method=POST \
  --oidc-service-account-email=invoker@project.iam.gserviceaccount.com
```

### Multi-Region Deployment

Deploy to multiple regions for geographic distribution:

```bash
for region in us-central1 europe-west1 asia-east1; do
  gcloud run deploy metamon-selfplay-$region \
    --image your-image \
    --region $region \
    --args="--num_battles=1000"
done
```

## See Also

- Main repository: [github.com/jakegrigsby/metamon](https://github.com/jakegrigsby/metamon)
- HuggingFace models: [huggingface.co/jakegrigsby/metamon](https://huggingface.co/jakegrigsby/metamon)
- Pokemon Showdown: [github.com/smogon/pokemon-showdown](https://github.com/smogon/pokemon-showdown)
