# RunPod: The Cheapest Cloud GPUs for Fine-Tuning

**Everything you need to know about using RunPod with tenfabric — pricing, setup, gotchas, and cost estimates for every model size.**

RunPod consistently offers the lowest spot GPU prices of any cloud provider that SkyPilot supports. This guide covers the RunPod-specific details that the general [infrastructure doc](infra-skypilot.md) doesn't — pricing comparisons, the full setup walkthrough, and the gotchas that will save you time.

---

## Why RunPod?

### Hard Numbers

Here's every GPU SkyPilot can provision, with spot pricing across all providers. Data from `config/defaults.py:67-74`:

| GPU | VRAM | RunPod Spot | AWS Spot | GCP Spot | Lambda | RunPod Savings vs AWS |
|-----|------|-------------|----------|----------|--------|----------------------|
| T4 | 16GB | **$0.16/hr** | $0.38/hr | $0.22/hr | — | 58% cheaper |
| L4 | 24GB | **$0.29/hr** | $0.48/hr | $0.35/hr | — | 40% cheaper |
| A10G | 24GB | **$0.44/hr** | $0.75/hr | $0.60/hr | — | 41% cheaper |
| A100-40GB | 40GB | **$1.24/hr** | $2.93/hr | $1.84/hr | $1.10/hr | 58% cheaper |
| A100-80GB | 80GB | **$1.64/hr** | $3.67/hr | $2.48/hr | $1.25/hr | 55% cheaper |
| H100 | 80GB | **$2.49/hr** | $6.50/hr | $4.76/hr | $1.99/hr | 62% cheaper |

RunPod is the cheapest option for T4, L4, A10G, and H100. Lambda is cheaper for A100s, but Lambda has limited availability and no spot market.

### Concrete Savings

Fine-tuning Llama-3.1-8B with LoRA 4-bit on an A10G:
- **RunPod:** $0.44/hr × 0.5 hr = **$0.22**
- **AWS:** $0.75/hr × 0.5 hr = **$0.38**
- **GCP:** $0.60/hr × 0.5 hr = **$0.30**

Over 100 training runs (common during experimentation): RunPod saves $16 vs AWS, $8 vs GCP.

### Tradeoffs

RunPod is not perfect for every use case:

| Advantage | Tradeoff |
|-----------|----------|
| Cheapest spot GPUs | Fewer regions than AWS/GCP |
| Simple pricing (no hidden fees) | Less enterprise tooling (no IAM, VPC, etc.) |
| Fast provisioning (~1 min) | Smaller GPU fleet — spot availability can be hit-or-miss |
| No minimum commitment | No SLA guarantees |
| GPU-focused infrastructure | Limited non-GPU instance types |

**Bottom line:** RunPod is ideal for individual developers and small teams doing fine-tuning. If you need enterprise compliance, multi-region redundancy, or complex networking, use AWS or GCP.

---

## Complete Setup Walkthrough

Six steps from zero to running. This is the most detailed walkthrough — follow it exactly and you'll avoid the common pitfalls.

### Step 1: Create a RunPod Account

Sign up at runpod.io. Add a payment method (credit card or crypto). You'll only be charged for actual GPU time — no upfront costs.

### Step 2: Generate an API Key

1. Go to your RunPod dashboard
2. Navigate to **Settings** → **API Keys**
3. Click **Create API Key**
4. Copy the key — you'll need it in Step 5

### Step 3: Set Up SSH Keys

**This is the #1 failure mode for new users.** SkyPilot connects to RunPod pods via SSH. Without an SSH key registered on RunPod, SkyPilot can create the pod but cannot connect to it, and the launch fails with a confusing timeout error.

**Why does SkyPilot need SSH?**

SkyPilot doesn't use RunPod's API to execute commands. Instead, it:
1. Creates a pod via the RunPod API (using your API key)
2. SSHes into the pod to upload files and run commands
3. Streams the output back to your terminal

This SSH-based approach gives SkyPilot a consistent execution model across all cloud providers (AWS, GCP, RunPod all use SSH).

**Generate a key if you don't have one:**

```bash
# Check if you already have a key
ls ~/.ssh/id_ed25519.pub

# If not, generate one (just press Enter for all prompts)
ssh-keygen -t ed25519
```

**Add the key to RunPod:**

1. Copy your public key:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
2. Go to RunPod dashboard → **Settings** → **SSH Public Keys**
3. Paste the entire contents of the file (starts with `ssh-ed25519 ...`)
4. Save

**Verify:** The key should appear in your SSH Public Keys list. If you have multiple keys (e.g., `id_rsa.pub` and `id_ed25519.pub`), add whichever SkyPilot will use. By default, SkyPilot tries `~/.ssh/id_rsa` then `~/.ssh/id_ed25519`.

### Step 4: Install SkyPilot with RunPod Support

```bash
uv pip install "skypilot[runpod]"
```

This installs SkyPilot with the RunPod provider plugin. The `[runpod]` extra pulls in the RunPod Python SDK.

### Step 5: Export Your API Key

```bash
export RUNPOD_API_KEY=your_key_here
```

SkyPilot reads this environment variable when communicating with RunPod's API.

**Make it persistent** by adding to your shell profile:

```bash
# For zsh (macOS default)
echo 'export RUNPOD_API_KEY=your_key_here' >> ~/.zshrc

# For bash
echo 'export RUNPOD_API_KEY=your_key_here' >> ~/.bashrc
```

### Step 6: Verify the Setup

```bash
sky check runpod
```

You should see output like:

```
Checking credentials to enable clouds for SkyPilot.
  RunPod: enabled
```

If it says "disabled", double-check:
- `RUNPOD_API_KEY` is exported in the current shell
- The API key is valid (not expired, not revoked)
- You installed `skypilot[runpod]`, not just `skypilot`

---

## What Happens When You Run `tfab train` with RunPod

Here's the RunPod-specific sequence with timing estimates for a small job (Qwen2.5-0.5B, 500 samples, 1 epoch):

### Timing Breakdown

```
Time    Event
─────   ──────────────────────────────────────────
0:00    tfab validates config, shows execution plan
0:05    SkyPilot generates task YAML
0:10    SkyPilot calls RunPod API to create pod
0:30    Pod is ready (RunPod provisions GPU instance)
0:45    SkyPilot SSH connects to pod
1:00    Setup: pip install tenfabric[training] from GitHub
2:30    Setup complete. Run command starts.
2:35    tfab train --local: validate config
2:40    Load dataset (500 samples from HuggingFace)
3:00    Download model weights (~1GB for 0.5B model)
3:30    Prepare model (quantize + attach LoRA)
3:40    Training starts (125 steps at bs=4)
5:00    Training complete
5:10    Export (merge adapters)
5:15    Done. sky launch returns on your laptop.
```

**Total wall time: ~5 minutes.** Of that, ~2 minutes is GPU training. The rest is infrastructure overhead (pod creation, SSH, package installation, model download).

### RunPod-Specific Details

**Pod creation (not EC2):** RunPod uses "pods" — lightweight GPU instances that boot faster than traditional VMs. A pod is essentially a Docker container with GPU passthrough. Provisioning typically takes 15-30 seconds.

**SSH connection:** After the pod is ready, SkyPilot establishes an SSH tunnel. This takes 5-15 seconds. The tunnel persists for the duration of the job, streaming stdout/stderr back to your terminal.

**File upload:** SkyPilot uses `rsync` over SSH to upload the re-entry config file to `/tmp/tenfabric-config.yaml`. For a single YAML file, this is nearly instant.

**Setup phase:** The `pip install` from GitHub is the slowest part of setup. It downloads and builds tenfabric plus all training dependencies (torch, transformers, etc.). This takes 60-90 seconds depending on the pod's internet speed.

**Training phase:** Identical to local training. The `LocalPipeline` runs the same 6 steps described in the [training pipeline doc](training-pipeline.md).

---

## Pricing Calculator

Estimated costs for common fine-tuning scenarios. All assume LoRA 4-bit, 1 epoch, 1000 samples, RunPod spot pricing.

| Model | Size | Recommended GPU | Spot $/hr | Est. Training Time | Est. Total Cost |
|-------|------|----------------|-----------|-------------------|----------------|
| Qwen2.5-0.5B | 0.5B | T4 (16GB) | $0.16 | ~3 min | **$0.01** |
| Llama-3.2-1B | 1B | T4 (16GB) | $0.16 | ~5 min | **$0.01** |
| Llama-3.2-3B | 3B | L4 (24GB) | $0.29 | ~10 min | **$0.05** |
| Mistral-7B | 7B | A10G (24GB) | $0.44 | ~20 min | **$0.15** |
| Llama-3.1-8B | 8B | A10G (24GB) | $0.44 | ~25 min | **$0.18** |
| Qwen2.5-14B | 14B | A100-40GB | $1.24 | ~40 min | **$0.83** |
| Qwen2.5-32B | 32B | A100-80GB | $1.64 | ~60 min | **$1.64** |
| Llama-3-70B | 70B | A100-80GB | $1.64 | ~120 min | **$3.28** |

**Notes:**
- Total cost includes ~3-5 min infrastructure overhead (pod creation, setup, model download)
- Training time scales roughly linearly with dataset size (2000 samples ≈ 2x the time)
- More epochs multiply training time accordingly (3 epochs ≈ 3x)
- Batch size affects training time: larger batches = fewer steps = faster (but more VRAM)

### Cost Optimization Tips

1. **Use spot instances** (`spot: true`) — 40-60% cheaper than on-demand
2. **Set `max_samples`** — Start with 500 for testing, scale up only when results look good
3. **Use 1 epoch first** — Fine-tune for 1 epoch, evaluate, then decide if more is needed
4. **Use `autostop`** — Don't let idle pods run. `10m` is plenty for most jobs
5. **Use the smallest GPU that fits** — Set `gpu: auto` to let tenfabric pick

---

## RunPod-Specific Gotchas

### 1. SSH Key is Mandatory

**Symptom:** `sky launch` creates a pod (visible in RunPod dashboard) but then hangs or fails with a timeout.

**Cause:** SkyPilot cannot SSH into the pod because no SSH key is registered.

**Fix:** Add your SSH public key to RunPod dashboard → Settings → SSH Public Keys. See the [setup walkthrough](#step-3-set-up-ssh-keys) above.

**Cleanup:** The orphaned pod is still running and billing. Go to the RunPod dashboard and terminate it manually, then retry.

### 2. Spot Availability

**Symptom:** `sky launch` fails with "No resources available" or takes a very long time.

**Cause:** No spot GPUs of the requested type are available. RunPod's GPU fleet is smaller than AWS/GCP, so popular GPUs (RTX 4090, A100) can run out.

**Fix options:**
- Wait and retry (availability fluctuates throughout the day)
- Try a different GPU type: `gpu: A10G` instead of `RTX4090`
- Switch to on-demand: `spot: false` (more expensive but guaranteed)
- Use a different provider: `provider: gcp` or `provider: aws`

### 3. Orphaned Pods

**Symptom:** After a failed launch or cancelled run, pods keep running in the RunPod dashboard.

**Cause:** SkyPilot may not clean up if the process is killed mid-launch (Ctrl+C, network drop, etc.).

**Fix:**
```bash
# Check for running clusters
sky status

# Terminate a specific cluster
sky down cloud-runpod

# Or terminate in RunPod dashboard directly
```

**Prevention:** Always use `autostop` (default is `30m`). Even if you forget to clean up, the pod will self-terminate.

### 4. Region Naming

**Symptom:** Setting `region: us-east-1` fails or selects the wrong location.

**Cause:** RunPod region names don't follow AWS naming conventions. RunPod uses its own region identifiers.

**Fix:** Leave `region: auto` (the default) and let SkyPilot handle region selection. If you need a specific region, check RunPod's documentation for their region identifiers.

### 5. Large Model Downloads

**Symptom:** Setup phase takes 10+ minutes for large models.

**Cause:** The model weights are downloaded from HuggingFace Hub to the pod every time. A 70B model in 4-bit is ~35GB.

**Fix:** For repeated runs with the same model, consider:
- Using SkyPilot file mounts to cache model weights on cloud storage
- Using Unsloth's pre-quantized models (smaller downloads)
- Keeping the pod alive between runs (set `autostop: never` during experimentation, but remember to clean up)

### 6. Disk Space

**Symptom:** Training fails with "No space left on device."

**Cause:** The default `disk_size: 100` may not be enough for very large models (70B+), which need space for weights, optimizer states, and checkpoints.

**Fix:** Increase disk size in your config:
```yaml
infra:
  disk_size: 200  # 200GB for 70B models
```

---

## Monitoring and Logs

### SkyPilot Commands

```bash
# Check cluster status
sky status                      # All clusters
sky status cloud-runpod         # Specific cluster

# Stream live logs from the pod
sky logs cloud-runpod           # Attaches to running job

# SSH into the pod
sky ssh cloud-runpod            # Interactive shell on the pod
```

### tenfabric Status

```bash
# View all training runs
uv run tfab status

# Output:
# ┌─────────┬───────────────┬──────────────────┬──────────┬───────────┐
# │ Run ID  │ Project       │ Model            │ Provider │ Status    │
# ├─────────┼───────────────┼──────────────────┼──────────┼───────────┤
# │ run-... │ cloud-runpod  │ Qwen/Qwen2.5-0.5B│ runpod  │ completed │
# └─────────┴───────────────┴──────────────────┴──────────┴───────────┘
```

### RunPod Dashboard

The RunPod web dashboard shows:
- Active pods and their GPU allocation
- Real-time billing
- Pod logs (if you prefer a web UI)
- SSH connection details (if you want to connect manually)

### Retrieving Outputs

Training outputs live on the pod. Three ways to get them:

**Option 1: Download via SkyPilot**
```bash
sky rsync-down cloud-runpod ./outputs/cloud-runpod/ ./local-outputs/
```

**Option 2: Push to HuggingFace Hub**
```yaml
output:
  push_to_hub: true
  hub_repo: your-username/my-model

infra:
  skypilot:
    envs:
      HF_TOKEN: hf_your_token
```

**Option 3: Mount cloud storage**
```yaml
infra:
  skypilot:
    file_mounts:
      /outputs: s3://my-bucket/training-outputs/
```

---

## Troubleshooting Quick Reference

| Error | Cause | Fix |
|-------|-------|-----|
| `SkyPilot not installed` | Missing cloud deps | `uv pip install "skypilot[runpod]"` |
| SSH timeout / pod created but can't connect | No SSH key on RunPod | Add `~/.ssh/id_ed25519.pub` to RunPod dashboard → Settings → SSH Keys |
| `sky check` shows RunPod disabled | API key not set | `export RUNPOD_API_KEY=your_key` |
| No GPUs available | Spot fleet exhausted | Try different GPU, use `spot: false`, or try different provider |
| Spot preemption mid-training | Another user outbid | Use `spot: false` for long jobs, or implement checkpointing |
| Disk full | Model + checkpoints too large | Increase `disk_size` in config |
| Pod still running after failure | Autostop hasn't kicked in | `sky down cloud-runpod` |

For additional troubleshooting, see the [Cloud RunPod example guide](examples/06-cloud-runpod/guide.md#troubleshooting).

---

## Further Reading

- **[Training Pipeline Deep Dive](training-pipeline.md)** — How the 6-step training pipeline works
- **[Infrastructure Architecture](infra-skypilot.md)** — How tenfabric uses SkyPilot for multi-cloud provisioning
- **[Cloud RunPod Example](examples/06-cloud-runpod/guide.md)** — Hands-on walkthrough with step-by-step commands
