# Cloud Fine-Tuning on RunPod: The Complete Guide

**What this teaches:** How tenfabric provisions a cloud GPU, runs training on it, and returns results — all from a single `tenfabric.yaml`. You'll understand the full lifecycle, not just the commands.

**Who it's for:** You've run Examples 01–05 locally and want to move to the cloud. You know what LoRA and SFT are. You want to understand what happens when you type `tfab train` with a cloud provider.

**Estimated cost:** $0.01–0.05 (spot RTX 4090 for ~2 minutes of GPU time).

---

## Prerequisites

1. **RunPod account** — Sign up at [runpod.io](https://runpod.io). Add a payment method.

2. **API key** — Generate one in your RunPod dashboard under Settings → API Keys.

3. **SSH public key on RunPod** — SkyPilot connects to pods via SSH. You must add your public key to RunPod:
   ```bash
   # Generate a key if you don't have one
   ssh-keygen -t ed25519
   ```
   Then go to RunPod dashboard → **Settings → SSH Public Keys** → paste the contents of `~/.ssh/id_ed25519.pub`. Without this, SkyPilot can create the pod but cannot connect to it, and the launch will fail.

4. **Install cloud dependencies:**
   ```bash
   uv pip install "skypilot[runpod]"
   ```

5. **Export your API key:**
   ```bash
   export RUNPOD_API_KEY=your_key_here
   ```

6. **Verify the setup:**
   ```bash
   sky check runpod
   ```
   You should see a green checkmark next to RunPod.

---

## Config Diff: Local vs Cloud

The entire model, dataset, training, and LoRA configuration is identical to Example 01. Only the `infra:` block changes:

```diff
 infra:
-  provider: local
+  provider: runpod
+  gpu: RTX4090
+  spot: true
+  autostop: 10m
+  disk_size: 50
```

That's it. Five lines turn a local training job into a cloud training job. Everything above `infra:` stays the same.

---

## The Big Picture

Here's the full lifecycle when you run `uv run tfab train tenfabric.yaml` with `provider: runpod`:

```
YOUR LAPTOP (control plane)                    RUNPOD VM (data plane)
─────────────────────────────                  ──────────────────────

1. Load tenfabric.yaml
2. Pydantic validates config
3. Show execution plan
4. Generate re-entry config ─────────┐
   (provider: local forced)          │
5. Generate SkyPilot YAML            │
6. sky launch ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┼─ ─ ─▶  7.  VM provisioned (RTX 4090)
   (blocks, waits for completion)    │         8.  pip install tenfabric[training] (from GitHub)
                                     │         9.  tfab train /tmp/config.yaml --local
                                     │              ├─ Step 1/6: Validate config
                                     │              ├─ Step 2/6: Detect GPU (RTX 4090, 24GB)
                                     │              ├─ Step 3/6: Load dataset (500 samples)
                                     │              ├─ Step 4/6: Prepare model (4-bit + LoRA)
                                     │              ├─ Step 5/6: SFT training
                                     │              └─ Step 6/6: Export (merge adapters)
                                     │        10.  Training complete
                                     └─────▶ 11.  VM auto-stops after 10m idle
12. sky launch returns
13. RunStore updated → completed
```

Two machines, one config. Your laptop is the **control plane** (orchestrates). The cloud VM is the **data plane** (does the work). They never run at the same time — `sky launch` blocks until the VM finishes.

---

## Phase-by-Phase Deep Dive

### Phase 1: Config Validation (your laptop)

```bash
uv run tfab train docs/examples/06-cloud-runpod/tenfabric.yaml
```

The CLI loads the YAML and passes it through Pydantic's `TenfabricConfig` model. This validates every field — types, ranges, enum values, cross-field rules (e.g., QLora requires quantization). If anything is wrong, you get a clear error before any cloud resources are touched.

Then tenfabric prints the execution plan:

```
╭──────────── Execution Plan ────────────╮
│ Project:  cloud-runpod                 │
│ Model:    Qwen/Qwen2.5-0.5B (lora, 4bit) │
│ Dataset:  tatsu-lab/alpaca (alpaca)    │
│ Training: sft — 1 epochs, bs=4, lr=0.0002 │
│ Infra:    runpod (gpu=RTX4090)         │
│ Est VRAM: ~2GB                         │
│ Suitable: RTX 3070, RTX 4060, ...      │
╰────────────────────────────────────────╯
```

Since `provider` is not `local`, tenfabric takes the cloud path.

**Source:** `cli/train.py` → `_show_plan()` then `_run_cloud()`

### Phase 2: SkyPilot YAML Generation (your laptop)

tenfabric translates your config into a SkyPilot task YAML. Here's what gets generated for this example:

```yaml
# ~/.tenfabric/skypilot/cloud-runpod.yaml (auto-generated)
name: cloud-runpod

resources:
  accelerators: RTX4090:1
  use_spot: true
  cloud: runpod
  disk_size: 50

setup: pip install 'tenfabric[training] @ git+https://github.com/billaram/tenfabric.git'

run: tfab train /tmp/tenfabric-config.yaml --local

file_mounts:
  /tmp/tenfabric-config.yaml: /Users/you/.tenfabric/skypilot/cloud-runpod-config.yaml
```

Field-by-field:

| SkyPilot Field | Comes From | Purpose |
|---|---|---|
| `name` | `config.project` | Cluster name for `sky status` / `sky down` |
| `accelerators` | `config.infra.gpu` + `gpu_count` | GPU type and count |
| `use_spot` | `config.infra.spot` | Spot instance for cheaper pricing |
| `cloud` | `config.infra.provider` → `PROVIDER_TO_SKY_CLOUD` map | Target cloud |
| `disk_size` | `config.infra.disk_size` | Storage for model weights + outputs |
| `setup` | Hardcoded + `config.infra.skypilot.setup` | Install deps on VM |
| `run` | Hardcoded | The re-entry command (see Phase 3) |
| `file_mounts` | Re-entry config path + `config.infra.skypilot.file_mounts` | Upload config to VM |

If you'd set `gpu: auto`, tenfabric would call `_auto_select_gpu()` which estimates VRAM needs from model size and picks the cheapest suitable GPU (preference: L4 → A10G → T4 → A100-40GB → A100-80GB → H100).

**Source:** `infra/skypilot.py` → `_generate_sky_yaml()`

### Phase 3: Cloud Re-Entry Config (your laptop)

This is the key insight. Before launching the VM, tenfabric writes a **modified copy** of your config:

```python
# infra/skypilot.py → _write_tenfabric_config()
config_dict = config.model_dump(mode="json")
config_dict["infra"]["provider"] = "local"   # <-- THE CRITICAL LINE
```

The re-entry config is identical to your original config except `provider` is forced to `local`. This is saved to `~/.tenfabric/skypilot/cloud-runpod-config.yaml` and uploaded to the VM at `/tmp/tenfabric-config.yaml`.

**Why?** Without this, the VM would see `provider: runpod` and try to provision *another* VM, which would provision another, and so on — infinite recursion. Forcing `provider: local` tells the VM: "You ARE the cloud GPU. Just train locally."

This is the **cloud re-entry pattern**: your laptop runs `tfab train config.yaml` (cloud path), the VM runs `tfab train config.yaml --local` (local path). Same CLI, same code, different execution modes.

**Source:** `infra/skypilot.py` → `_write_tenfabric_config()`

### Phase 4: `sky launch` (your laptop → RunPod)

tenfabric runs:

```bash
sky launch ~/.tenfabric/skypilot/cloud-runpod.yaml -y --cluster cloud-runpod
```

SkyPilot takes over and does the heavy lifting:

1. **Finds the cheapest spot RTX 4090** on RunPod matching your constraints
2. **Provisions the VM** — creates the instance, attaches the GPU, sets up SSH
3. **Uploads file mounts** — copies the re-entry config to `/tmp/tenfabric-config.yaml`
4. **Runs the `setup` script** — installs `tenfabric[training]` (see Phase 5)
5. **Runs the `run` command** — executes `tfab train /tmp/tenfabric-config.yaml --local` (see Phase 6)
6. **Streams logs** back to your terminal
7. **Blocks** until the run command completes (or the 600s timeout is hit)

If the launch fails (no GPUs available, auth error, timeout), tenfabric raises a `RuntimeError` with the stderr output.

**Source:** `infra/skypilot.py` → `SkyPilotProvider.provision()`

### Phase 5: VM Setup (on RunPod)

The `setup` command runs once after the VM is created:

```bash
pip install 'tenfabric[training] @ git+https://github.com/billaram/tenfabric.git'
```

Since tenfabric is not yet published to PyPI, the VM installs directly from GitHub. This pulls in the `[training]` extra group from `pyproject.toml`:
- **torch** — The ML framework
- **transformers** — Model loading (AutoModelForCausalLM, AutoTokenizer)
- **datasets** — Dataset loading from HuggingFace Hub
- **trl** — Training library (SFTTrainer, DPOTrainer)
- **peft** — LoRA adapter support
- **bitsandbytes** — 4-bit/8-bit quantization

If you'd set `backend: unsloth`, the setup would also install the `[unsloth]` extra from GitHub.

### Phase 6: Training (on RunPod)

Now the VM runs:

```bash
tfab train /tmp/tenfabric-config.yaml --local
```

The `--local` flag forces `provider: local` (and the re-entry config already has `provider: local`), so tenfabric enters the `LocalPipeline`. This is the exact same 6-step flow as running locally on your own GPU:

| Step | What Happens | Key Code |
|------|-------------|----------|
| 1. Validate config | Pydantic already validated; runtime checks | `pipeline.py` → `_validate()` |
| 2. Detect GPU | `torch.cuda.get_device_name(0)` → "NVIDIA RTX 4090 (24GB)" | `pipeline.py` → `_detect_gpu()` |
| 3. Load dataset | Downloads `tatsu-lab/alpaca`, selects 500 samples | `pipeline.py` → `_load_dataset()` |
| 4. Prepare model | Loads Qwen2.5-0.5B in 4-bit, attaches LoRA adapters (~1.5M trainable params) | `trl_backend.py` → `prepare_model()` |
| 5. Train | SFTTrainer runs 1 epoch (125 steps at bs=4 over 500 samples) | `trl_backend.py` → `_train_sft()` |
| 6. Export | Merges LoRA adapters back into base model | `export.py` → `export_model()` |

The model preparation step (step 4) is worth understanding:
1. Load base model with `BitsAndBytesConfig(load_in_4bit=True)` — shrinks 0.5B params to ~0.5GB
2. Call `prepare_model_for_kbit_training()` — enables gradient computation on quantized model
3. Auto-detect LoRA target modules (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
4. Attach LoRA adapters with `get_peft_model()` — adds ~1.5M trainable params

**Source:** `core/pipeline.py` → `LocalPipeline.run()`

### Phase 7: Completion & Autostop

After training finishes:

1. `sky launch` **returns** on your laptop (it was blocking this whole time)
2. tenfabric updates **RunStore** (`~/.tenfabric/runs.db`) with `status: completed`
3. The VM sits **idle** — no process running
4. After 10 minutes idle (your `autostop: 10m` setting), SkyPilot **auto-stops** the VM
5. Auto-stop means the VM is stopped, not deleted. It won't charge GPU time but may charge minimal storage.

To fully remove the cluster and stop all charges:

```bash
sky down cloud-runpod
```

---

## Files Created

| File | Location | Purpose |
|------|----------|---------|
| Re-entry config | `~/.tenfabric/skypilot/cloud-runpod-config.yaml` | Config with `provider: local` for VM |
| SkyPilot YAML | `~/.tenfabric/skypilot/cloud-runpod.yaml` | Task definition for `sky launch` |
| Run history | `~/.tenfabric/runs.db` | SQLite DB tracking all runs |
| Model outputs | `./outputs/cloud-runpod/` (on VM) | Trained adapter + merged model |

---

## Cost Breakdown

RunPod has the cheapest spot GPU prices. Here's what SkyPilot can access:

| GPU | VRAM | RunPod Spot | AWS Spot | GCP Spot |
|-----|------|-------------|----------|----------|
| T4 | 16GB | $0.16/hr | $0.38/hr | $0.22/hr |
| L4 | 24GB | $0.29/hr | $0.48/hr | $0.35/hr |
| A10G | 24GB | $0.44/hr | $0.75/hr | $0.60/hr |
| A100-40GB | 40GB | $1.24/hr | $2.93/hr | $1.84/hr |
| A100-80GB | 80GB | $1.64/hr | $3.67/hr | $2.48/hr |
| H100 | 80GB | $2.49/hr | $6.50/hr | $4.76/hr |

**This example's cost:** RTX 4090 spot on RunPod is ~$0.39/hr. Training takes ~2 minutes of GPU time, plus ~3 minutes for setup (downloading model, installing deps). Total wall time: ~5 minutes = **~$0.03**.

---

## Retrieving Outputs

Training outputs live on the VM. Three ways to get them:

### Option 1: `sky rsync-down` (manual download)

```bash
# Download the merged model
sky rsync-down cloud-runpod ./outputs/cloud-runpod/ ./local-outputs/

# Download just the adapter (smaller)
sky rsync-down cloud-runpod ./outputs/cloud-runpod/adapter_model.safetensors ./local-outputs/
```

### Option 2: Push to HuggingFace Hub (automated)

Add to your config:

```yaml
output:
  dir: ./outputs/cloud-runpod
  merge_adapter: true
  push_to_hub: true
  hub_repo: your-username/cloud-runpod-qwen
```

The VM pushes the model directly to HuggingFace Hub at the end of training. No manual download needed. Make sure `HF_TOKEN` is set — you can pass it via the SkyPilot envs:

```yaml
infra:
  skypilot:
    envs:
      HF_TOKEN: hf_your_token_here
```

### Option 3: Mount cloud storage

For production workflows, mount S3/GCS via SkyPilot file mounts so outputs write directly to cloud storage:

```yaml
infra:
  skypilot:
    file_mounts:
      /outputs: s3://my-bucket/training-outputs/
```

---

## Scaling Up

### Larger models (7B–13B)

```yaml
model:
  base: meta-llama/Llama-3.1-8B    # 8B params, ~12GB VRAM with 4-bit LoRA

infra:
  gpu: A10G                          # 24GB VRAM, $0.44/hr spot on RunPod
  disk_size: 100                     # More space for larger weights
```

### Even larger (34B–70B)

```yaml
model:
  base: Qwen/Qwen2.5-72B

infra:
  gpu: A100-80GB                     # 80GB VRAM, $1.64/hr spot on RunPod
  disk_size: 200
```

### Let SkyPilot pick the GPU

```yaml
infra:
  gpu: auto    # tenfabric estimates VRAM, picks cheapest GPU that fits
```

`auto` mode uses the VRAM estimation tables in `config/defaults.py`. It calls `guess_model_size()` → `estimate_vram()` → `recommend_gpu()`, then picks from a cloud-preference order: L4 → A10G → T4 → A100-40GB → A100-80GB → H100.

### Multi-GPU

```yaml
infra:
  gpu: A100-80GB
  gpu_count: 2    # SkyPilot YAML becomes "accelerators: A100-80GB:2"
```

---

## Monitoring

### Check cluster status
```bash
sky status                  # All clusters
sky status cloud-runpod     # Specific cluster
```

### View training logs
```bash
sky logs cloud-runpod       # Stream logs from the VM
```

### Check run history
```bash
uv run tfab status          # All runs from ~/.tenfabric/runs.db
```

---

## Troubleshooting

### "SkyPilot not installed"
```
RuntimeError: SkyPilot not installed. Install cloud dependencies:
  pip install 'tenfabric[cloud]'
```
Install the cloud extra: `uv pip install "skypilot[runpod]"`

### SSH connection failed / pod created but launch fails
The pod appears in the RunPod dashboard but `sky launch` fails. This means SkyPilot cannot SSH into the pod. Fix:
1. Check you have an SSH key: `ls ~/.ssh/id_ed25519.pub` (or `id_rsa.pub`)
2. Add the public key to RunPod dashboard → **Settings → SSH Public Keys**
3. Terminate the orphaned pod in the RunPod dashboard, then retry

### "No API key" / sky check fails
```bash
export RUNPOD_API_KEY=your_key_here
sky check runpod
```
Make sure the key is exported in the shell where you run `tfab train`.

### Spot preemption
If your spot instance gets reclaimed mid-training, `sky launch` will return a non-zero exit code, and tenfabric reports the run as failed. For longer jobs where preemption is a concern:

```yaml
infra:
  spot: false    # Use on-demand instances (2–3x more expensive, but guaranteed)
```

### Timeout
`sky launch` has a 600-second (10 minute) timeout in tenfabric. If setup + training takes longer, the process will be killed. For larger jobs, this timeout may need adjustment in `infra/skypilot.py`.

### Forgotten clusters
Stopped clusters still exist and may incur minimal storage charges. Clean up:

```bash
sky status          # See all clusters
sky down cloud-runpod   # Delete a specific cluster
sky down -a         # Delete ALL clusters (careful!)
```

### "Training failed" but cluster is still running
If training errors out, the VM may still be running (autostop hasn't kicked in yet). Manually stop it:

```bash
sky down cloud-runpod
```

---

## Source Code Map

For developers who want to trace the code path:

```
tfab train tenfabric.yaml
│
├─ cli/train.py → train_cmd()
│  ├─ load_config()                          # YAML → TenfabricConfig (Pydantic)
│  ├─ _show_plan()                           # Rich Panel with execution plan
│  └─ _run_cloud()                           # provider != local → cloud path
│     └─ CloudPipeline().run()               # core/pipeline.py
│        └─ SkyPilotProvider().provision()    # infra/skypilot.py
│           ├─ _write_tenfabric_config()     # Serialize config, force provider: local
│           ├─ _generate_sky_yaml()          # Build SkyPilot task dict
│           ├─ _write_sky_yaml()             # Write to ~/.tenfabric/skypilot/
│           └─ subprocess: sky launch        # Blocks until VM finishes
│
│  ──── ON THE VM ────
│
├─ cli/train.py → train_cmd(local=True)
│  ├─ load_config()                          # Same validation
│  ├─ _show_plan()                           # Same plan, but provider=local now
│  └─ _run_local()                           # --local flag → local path
│     └─ LocalPipeline().run()               # core/pipeline.py
│        ├─ _validate()                      # Runtime checks
│        ├─ _detect_gpu()                    # torch.cuda → RTX 4090 (24GB)
│        ├─ _load_dataset()                  # datasets.load_dataset()
│        ├─ _prepare_model()                 # trl_backend.prepare_model()
│        │  ├─ BitsAndBytesConfig(4bit)      # Quantize base model
│        │  ├─ AutoModelForCausalLM          # Load Qwen2.5-0.5B
│        │  ├─ prepare_model_for_kbit        # Enable gradients on quantized model
│        │  └─ get_peft_model(LoraConfig)    # Attach LoRA adapters
│        ├─ _train()                         # trl_backend._train_sft()
│        │  └─ SFTTrainer.train()            # 1 epoch, 125 steps
│        └─ _export()                        # export.export_model()
│           └─ merge_and_unload()            # LoRA → merged model
```
