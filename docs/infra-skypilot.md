# How tenfabric Uses SkyPilot for Cloud GPU Provisioning

**An architecture deep dive for developers.**

After reading this, you'll understand:

- How tenfabric abstracts infrastructure behind the `InfraProvider` Protocol
- The **cloud re-entry pattern** that lets the same CLI run on your laptop and on a cloud VM
- How SkyPilot YAML is generated from your `tenfabric.yaml` — field by field
- How the auto GPU selection algorithm picks the cheapest GPU that fits your model
- The three execution modes: Local, Cloud Direct, and Cloud Durable (Temporal)

**Prerequisites:** You've read the [training pipeline doc](training-pipeline.md) or are familiar with tenfabric's 6-step pipeline. You know what SkyPilot is (a multi-cloud launcher) at a high level.

---

## Why SkyPilot?

### The Multi-Cloud Problem

Cloud GPUs are expensive, and pricing varies wildly between providers:

| GPU | AWS Spot | GCP Spot | RunPod Spot | Lambda |
|-----|----------|----------|-------------|--------|
| T4 (16GB) | $0.38/hr | $0.22/hr | $0.16/hr | — |
| L4 (24GB) | $0.48/hr | $0.35/hr | $0.29/hr | — |
| A10G (24GB) | $0.75/hr | $0.60/hr | $0.44/hr | — |
| A100-40GB | $2.93/hr | $1.84/hr | $1.24/hr | $1.10/hr |
| A100-80GB | $3.67/hr | $2.48/hr | $1.64/hr | $1.25/hr |
| H100 (80GB) | $6.50/hr | $4.76/hr | $2.49/hr | $1.99/hr |

*(Source: `config/defaults.py:67-74`)*

Without SkyPilot, you'd need to learn each provider's CLI, API, instance types, and pricing. SkyPilot provides:

- **Unified YAML** — one task definition works across all clouds
- **Spot management** — automatic fallback when spot instances are preempted
- **SSH execution** — runs commands on VMs, uploads/downloads files
- **Cost optimization** — finds the cheapest available option

### Provider Mapping

From `infra/skypilot.py:20-26`:

```python
PROVIDER_TO_SKY_CLOUD: dict[str, str] = {
    "aws": "aws",
    "gcp": "gcp",
    "azure": "azure",
    "runpod": "runpod",
    "lambda": "lambda",
}
```

tenfabric's `InfraProvider` enum (`config/schema.py:39-46`) includes `auto`, `local`, `aws`, `gcp`, `azure`, `runpod`, and `lambda`. The `auto` and `local` values are handled specially — `auto` lets SkyPilot choose the cheapest cloud, and `local` bypasses SkyPilot entirely.

---

## The InfraProvider Protocol

### Protocol vs ABC

tenfabric uses Python's `Protocol` (structural typing) instead of an abstract base class (ABC). From `infra/base.py:25-42`:

```python
class InfraProvider(Protocol):
    """Protocol for infrastructure providers."""

    def provision(self, config: TenfabricConfig) -> InfraHandle:
        """Provision compute resources and return a handle."""
        ...

    def setup(self, handle: InfraHandle, config: TenfabricConfig) -> None:
        """Install dependencies and prepare the environment."""
        ...

    def teardown(self, handle: InfraHandle) -> None:
        """Release provisioned resources."""
        ...

    def status(self, handle: InfraHandle) -> str:
        """Check the status of provisioned infrastructure."""
        ...
```

**Why Protocol?** With a Protocol, any class that implements these four methods is a valid `InfraProvider` — no inheritance required. This makes it easy to add new providers (RunPod, Lambda, your custom on-prem system) without coupling to a base class. Duck typing, but with type checker support.

### InfraHandle Dataclass

From `infra/base.py:12-23`:

```python
@dataclass
class InfraHandle:
    provider: str                        # "local", "skypilot"
    instance_id: str | None = None       # Cluster name or "localhost"
    host: str | None = None              # SSH hostname
    port: int | None = None              # SSH port
    gpu_name: str | None = None          # Detected GPU name
    gpu_count: int = 1                   # Number of GPUs
    status: str = "pending"              # "pending", "ready", "stopped"
    metadata: dict = field(default_factory=dict)  # Provider-specific data
```

`InfraHandle` is the return type of `provision()`. It carries enough information for subsequent `setup()`, `teardown()`, and `status()` calls. The `metadata` dict allows providers to store extra information (e.g., SkyPilot stores the YAML path).

### Two Implementations

| Method | LocalProvider | SkyPilotProvider |
|--------|--------------|------------------|
| `provision()` | Detect local GPU via `torch.cuda` | Generate SkyPilot YAML, run `sky launch` |
| `setup()` | No-op (user manages env) | No-op (SkyPilot handles via `setup:` in YAML) |
| `teardown()` | No-op (nothing to release) | Run `sky down {cluster}` |
| `status()` | Always returns `"ready"` | Run `sky status {cluster} --format json` |

**LocalProvider** (`infra/local.py:9-32`) is trivial — it detects your GPU and returns a handle. No provisioning needed.

**SkyPilotProvider** (`infra/skypilot.py:29-94`) does the heavy lifting — it generates a SkyPilot task YAML, launches it via the CLI, and manages the cluster lifecycle.

---

## The Cloud Re-Entry Pattern

This is the architectural centerpiece of tenfabric's cloud execution. Understanding this pattern is essential.

### The Two-Machine Diagram

```
YOUR LAPTOP (control plane)                 CLOUD VM (data plane)
═══════════════════════════                 ═══════════════════════

tenfabric.yaml says:                        re-entry config says:
  provider: runpod                            provider: local    ← FORCED
  gpu: RTX4090                                gpu: RTX4090
  ...                                         ...

tfab train tenfabric.yaml                   tfab train /tmp/config.yaml --local
        │                                           │
        ├─ load_config()                            ├─ load_config()
        ├─ _show_plan()                             ├─ _show_plan()
        ├─ provider != local                        ├─ provider == local
        │  → _run_cloud()                           │  → _run_local()
        │                                           │
        ├─ CloudPipeline().run()                    ├─ LocalPipeline().run()
        │  ├─ _write_tenfabric_config()             │  ├─ Step 1: Validate
        │  │   └─ provider forced to "local"        │  ├─ Step 2: Detect GPU
        │  ├─ _generate_sky_yaml()                  │  ├─ Step 3: Load Dataset
        │  ├─ _write_sky_yaml()                     │  ├─ Step 4: Prepare Model
        │  └─ sky launch ─────────────────▶         │  ├─ Step 5: Train
        │     (blocks until VM finishes)            │  └─ Step 6: Export
        │                             ◀─────────────┘
        │     sky launch returns
        └─ RunStore: status=completed
```

### Why This Pattern Exists

The same `tfab` CLI must work in two contexts:

1. **On your laptop** — Orchestrate cloud provisioning, then wait
2. **On the VM** — Actually run the training

The cloud re-entry pattern achieves this with zero special-casing in the training code. The `LocalPipeline` doesn't know or care whether it's running on your laptop or a cloud VM. It just trains.

### How Infinite Recursion Is Prevented

From `infra/skypilot.py:109-119`:

```python
def _write_tenfabric_config(config: TenfabricConfig) -> Path:
    config_dict = config.model_dump(mode="json")
    config_dict["infra"]["provider"] = "local"  # ← THE CRITICAL LINE
    # ...write to ~/.tenfabric/skypilot/{project}-config.yaml
```

This is the key. When the config is serialized for the VM:

1. The original config says `provider: runpod`
2. The re-entry config forces `provider: local`
3. The SkyPilot YAML runs: `tfab train /tmp/tenfabric-config.yaml --local`

The `--local` flag in `cli/train.py:50-51` provides a second layer of protection:

```python
if local:
    cfg.infra.provider = InfraProvider.LOCAL
```

Without the `provider: local` override, the VM would see `provider: runpod`, try to provision another VM, which would try to provision another... infinite recursion, infinite cloud bill.

### The SkyPilot YAML Run Command

From `infra/skypilot.py:170`:

```python
sky_config["run"] = "tfab train /tmp/tenfabric-config.yaml --local"
```

This is what SkyPilot executes after setup completes. It invokes the exact same CLI, but with the modified config and the `--local` flag.

---

## SkyPilot YAML Generation — Field by Field

### Complete Annotated Example

For a config with `provider: runpod`, `gpu: RTX4090`, `spot: true`:

```yaml
# Generated by _generate_sky_yaml() in infra/skypilot.py:122-181
name: cloud-runpod                  # ← config.project

resources:
  accelerators: RTX4090:1           # ← config.infra.gpu : config.infra.gpu_count
  use_spot: true                    # ← config.infra.spot
  cloud: runpod                     # ← PROVIDER_TO_SKY_CLOUD[config.infra.provider.value]
  disk_size: 50                     # ← config.infra.disk_size

setup: |
  pip install 'tenfabric[training] @ git+https://github.com/billaram/tenfabric.git'

run: tfab train /tmp/tenfabric-config.yaml --local

file_mounts:
  /tmp/tenfabric-config.yaml: /Users/you/.tenfabric/skypilot/cloud-runpod-config.yaml
```

### Mapping Table

| SkyPilot Field | tenfabric Source | Notes |
|---------------|-----------------|-------|
| `name` | `config.project` | Used as cluster name for `sky status`/`sky down` |
| `resources.accelerators` | `config.infra.gpu` + `config.infra.gpu_count` | Format: `GPU_TYPE:COUNT`. If `gpu: auto`, calls `_auto_select_gpu()` |
| `resources.use_spot` | `config.infra.spot` | Spot = cheaper, but can be preempted |
| `resources.cloud` | `PROVIDER_TO_SKY_CLOUD[provider.value]` | Only set when provider != `auto`. Omitted for `auto` (SkyPilot picks cheapest) |
| `resources.region` | `config.infra.region` | Only set when region != `"auto"` |
| `resources.disk_size` | `config.infra.disk_size` | In GB. Default 100, this example uses 50 |
| `setup` | Hardcoded + `config.infra.skypilot.setup` | Installs tenfabric from GitHub. Adds Unsloth install if `backend: unsloth` |
| `run` | Hardcoded | Always `tfab train /tmp/tenfabric-config.yaml --local` |
| `file_mounts` | Re-entry config path + `config.infra.skypilot.file_mounts` | Uploads the modified config to the VM |
| `envs` | `config.infra.skypilot.envs` | Only set if user provides environment variables |

### Setup Command Construction

From `infra/skypilot.py:156-167`:

```python
pip_source = "git+https://github.com/billaram/tenfabric.git"
setup_lines = [
    f"pip install 'tenfabric[training] @ {pip_source}'",
]
if config.training.backend.value == "unsloth":
    setup_lines.append(f"pip install 'tenfabric[unsloth] @ {pip_source}'")

if config.infra.skypilot.setup:
    setup_lines.append(config.infra.skypilot.setup)

sky_config["setup"] = " && ".join(setup_lines)
```

The setup always installs `tenfabric[training]` (which includes torch, transformers, trl, peft, bitsandbytes, datasets). If the backend is Unsloth, it also installs the `[unsloth]` extra. User-provided setup commands (via `SkyPilotPassthrough`) are appended.

### SkyPilotPassthrough for Power Users

From `config/schema.py:113-118`:

```python
class SkyPilotPassthrough(BaseModel):
    file_mounts: dict[str, str] = Field(default_factory=dict)
    envs: dict[str, str] = Field(default_factory=dict)
    setup: str | None = None
```

This allows users to inject raw SkyPilot configuration:

```yaml
infra:
  provider: runpod
  skypilot:
    file_mounts:
      /data/my-dataset: s3://my-bucket/datasets/custom/
    envs:
      HF_TOKEN: hf_abc123
      WANDB_API_KEY: wandb_xyz
    setup: "pip install flash-attn --no-build-isolation"
```

These are merged into the generated SkyPilot YAML. File mounts are added to the default mount (the re-entry config). Environment variables are added as a top-level `envs` key. Custom setup is appended to the standard setup command with `&&`.

---

## Auto GPU Selection Algorithm

When you set `gpu: auto`, tenfabric picks the cheapest GPU that can handle your model. Here's the pipeline:

```
config.model.base               config.model.method    config.model.quantization
      │                                │                        │
      ▼                                │                        │
guess_model_size()                     │                        │
      │                                │                        │
      ▼                                ▼                        ▼
 model_size_b ──────────────▶ estimate_vram(size, method, quant)
                                       │
                                       ▼
                                  vram_needed
                                       │
                                       ▼
                              recommend_gpu(vram_needed)
                                       │
                                       ▼
                              suitable_gpus (sorted by VRAM)
                                       │
                                       ▼
                              cloud_preference filter
                              [L4, A10G, T4, A100-40GB, A100-80GB, H100]
                                       │
                                       ▼
                              selected_gpu (cheapest that fits)
```

### Step 1: Guess Model Size

From `config/defaults.py:77-83`:

```python
def guess_model_size(model_id: str) -> float | None:
    model_lower = model_id.lower()
    for key, size in MODEL_SIZES.items():
        if key in model_lower:
            return size
    return None
```

This does substring matching against a lookup table of known models (`config/defaults.py:19-43`). Examples:

| Model ID | Matched Key | Size |
|----------|-------------|------|
| `Qwen/Qwen2.5-0.5B` | `qwen2.5-0.5b` | 0.5B |
| `meta-llama/Llama-3.1-8B` | `llama-3.1-8b` | 8.0B |
| `mistralai/Mistral-7B-v0.1` | `mistral-7b` | 7.0B |
| `unknown/custom-model` | No match | `None` → defaults to `A10G` |

### Step 2: Estimate VRAM

From `config/defaults.py:86-107`:

```python
def estimate_vram(model_size_b: float, method: str, quantization: str) -> float:
    key = f"{method}_{quantization}" if method != "full" else "full"

    sizes = sorted(VRAM_ESTIMATES.keys())
    closest = min(sizes, key=lambda s: abs(s - model_size_b))

    estimates = VRAM_ESTIMATES.get(closest, {})
    base_estimate = estimates.get(key)

    ratio = model_size_b / closest if closest > 0 else 1
    return base_estimate * ratio
```

It finds the closest entry in the VRAM table and scales linearly. For a 9B model with `lora_4bit`, it would find the 8B entry (12GB) and scale: `12 × (9/8) = 13.5GB`.

### Step 3: Recommend GPUs

From `config/defaults.py:110-116`:

```python
def recommend_gpu(vram_needed: float) -> list[str]:
    suitable = []
    for gpu, vram in sorted(GPU_VRAM.items(), key=lambda x: x[1]):
        if vram >= vram_needed * 1.1:  # 10% headroom
            suitable.append(gpu)
    return suitable
```

Returns all GPUs with at least 10% more VRAM than needed, sorted by VRAM ascending. The 10% headroom accounts for PyTorch memory overhead, optimizer states, and activation memory.

### Step 4: Cloud Preference Ordering

From `infra/skypilot.py:206-210`:

```python
cloud_preference = ["L4", "A10G", "T4", "A100-40GB", "A100-80GB", "H100"]
for gpu in cloud_preference:
    if gpu in suitable:
        return gpu
return suitable[0] if suitable else "A10G"
```

This filters the suitable GPUs through a cost-preference order. L4 and A10G are generally the best price-performance for fine-tuning. If nothing in the preference list fits, it falls back to the first suitable GPU. If no GPU is suitable at all, it defaults to `A10G` (24GB).

### Worked Example: Llama-3.1-8B with QLoRA

```
Input:
  model.base = "meta-llama/Llama-3.1-8B"
  model.method = "qlora"
  model.quantization = "4bit"

Step 1: guess_model_size("meta-llama/Llama-3.1-8B")
        → matches "llama-3.1-8b" → 8.0B

Step 2: estimate_vram(8.0, "qlora", "4bit")
        → key = "qlora_4bit"
        → closest size = 8.0 in VRAM_ESTIMATES
        → VRAM_ESTIMATES[8.0]["qlora_4bit"] = 10GB
        → ratio = 8.0/8.0 = 1.0
        → 10 × 1.0 = 10GB

Step 3: recommend_gpu(10)
        → need 10 × 1.1 = 11GB minimum
        → RTX 3060 (12GB) ✓, RTX 4070 (12GB) ✓, RTX 3090 (24GB) ✓,
          RTX 4090 (24GB) ✓, T4 (16GB) ✓, A10G (24GB) ✓, ...

Step 4: cloud_preference filter
        → L4 (24GB) ✓ → selected!

Result: gpu = "L4"
```

---

## File Artifacts and Locations

### On Your Laptop

| File | Path | Created By | Purpose |
|------|------|-----------|---------|
| Re-entry config | `~/.tenfabric/skypilot/{project}-config.yaml` | `_write_tenfabric_config()` | Config with `provider: local` for VM upload |
| SkyPilot YAML | `~/.tenfabric/skypilot/{project}.yaml` | `_write_sky_yaml()` | Task definition passed to `sky launch` |
| Run history DB | `~/.tenfabric/runs.db` | `RunStore` | SQLite DB tracking all training runs |

### On the Cloud VM

| File | Path | Created By | Purpose |
|------|------|-----------|---------|
| Uploaded config | `/tmp/tenfabric-config.yaml` | SkyPilot file mount | Config the VM uses for training |
| Model outputs | `{config.output.dir}/` | Training + export | Adapter weights, merged model, GGUF |
| Checkpoints | `{config.output.dir}/checkpoint-*/` | SFTTrainer | Periodic saves during training |

---

## Teardown, Status, and Error Handling

### Teardown

From `infra/skypilot.py:63-77`:

```python
def teardown(self, handle: InfraHandle) -> None:
    cluster_name = handle.instance_id
    if not cluster_name:
        return

    try:
        subprocess.run(
            ["sky", "down", cluster_name, "-y"],
            capture_output=True, text=True, timeout=120,
        )
    except Exception as e:
        console.print(f"    Warning: Failed to tear down cluster: {e}")
        console.print(f"    Manual cleanup: sky down {cluster_name}")
```

Teardown calls `sky down` to destroy the cluster. The `-y` flag auto-confirms. If it fails (network issue, SkyPilot bug), it prints the manual cleanup command. The 120-second timeout prevents hanging.

Note: In the current `CloudPipeline`, teardown is not called automatically — the `autostop` setting handles it. The VM idles after training and auto-stops after the configured duration (e.g., `10m`). To fully delete:

```bash
sky down {cluster_name}
```

### Status

From `infra/skypilot.py:79-94`:

```python
def status(self, handle: InfraHandle) -> str:
    result = subprocess.run(
        ["sky", "status", cluster_name, "--format", "json"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode == 0:
        data = json.loads(result.stdout)
        if data:
            return data[0].get("status", "unknown")
    return "unknown"
```

Queries SkyPilot for the cluster's current state. Returns the status string from SkyPilot's JSON output (e.g., "UP", "STOPPED", "INIT").

### Error Handling

From `infra/skypilot.py:97-106`:

```python
def _check_skypilot_installed() -> None:
    try:
        import sky
    except ImportError:
        raise RuntimeError(
            "SkyPilot not installed. Install cloud dependencies:\n"
            "  pip install 'tenfabric[cloud]'\n\n"
            "Then configure your cloud credentials:\n"
            "  sky check"
        )
```

This is called at the start of `provision()`. If SkyPilot isn't installed, the user gets a clear error message with install instructions — before any cloud resources are touched.

Launch failures (`sky launch` returns non-zero):

```python
if result.returncode != 0:
    raise RuntimeError(f"SkyPilot launch failed:\n{result.stderr}")
```

The stderr output is captured and included in the error message. Common failures:
- No matching GPU available (try a different GPU type or on-demand instead of spot)
- Authentication error (run `sky check` to verify credentials)
- SSH connection failure (ensure SSH key is added to the cloud provider)

---

## Three Execution Modes

tenfabric supports three ways to run training, selected by the config and CLI flags:

```
                        tfab train config.yaml
                               │
                    ┌──────────┼──────────────┐
                    │          │              │
              provider=local   │        temporal_address set
              or --local flag  │              │
                    │          │              │
                    ▼          ▼              ▼
              ┌──────────┐ ┌──────────┐ ┌──────────────┐
              │  Local   │ │  Cloud   │ │   Temporal    │
              │ Pipeline │ │ Pipeline │ │   Pipeline    │
              └──────────┘ └──────────┘ └──────────────┘
                    │          │              │
                    ▼          ▼              ▼
              Run 6 steps    SkyPilot      Temporal
              in-process     provisions    workflow with
                             VM, runs      7 activities,
                             tfab --local  retries, and
                             on VM         durability
```

### Mode 1: Local (LocalPipeline)

**When:** `provider: local` or `--local` flag.

**What:** Runs all 6 training steps sequentially in the current process. No infrastructure provisioning. Uses whatever GPU is available on the current machine.

**Code path:** `cli/train.py:70-71` → `_run_local()` → `LocalPipeline().run()`

**Best for:** Development, testing, machines with their own GPU.

### Mode 2: Cloud Direct (CloudPipeline)

**When:** `provider` is a cloud (aws, gcp, runpod, etc.) and `workflow.temporal_address` is empty.

**What:** Generates SkyPilot YAML, runs `sky launch`, blocks until the VM finishes training. No Temporal involved — simple subprocess execution.

**Code path:** `cli/train.py:73` → `_run_cloud()` → `CloudPipeline().run()` → `SkyPilotProvider().provision()`

**Best for:** One-off cloud training jobs. Simple, no infrastructure dependencies.

### Mode 3: Cloud Durable (TemporalPipeline)

**When:** `workflow.temporal_address` is set (or empty — auto-starts dev server).

**What:** Runs training as a Temporal workflow with 7 activities. Temporal provides:
- **Durability** — workflow state survives process crashes
- **Retries** — failed activities are retried with configurable backoff
- **Visibility** — workflow status visible in Temporal UI

**Code path:** `cli/train.py:119-123` → `_run_cloud()` → `TemporalPipeline().run()` → `start_training_workflow()`

From `workflows/client.py:18-33`:

```python
async def start_training_workflow(run_id, config, temporal_address=None):
    address = temporal_address or config.workflow.temporal_address
    task_queue = config.workflow.task_queue or DEFAULT_TASK_QUEUE

    if address:
        return await _run_with_external_server(address, run_id, config_dict, task_queue)
    else:
        return await _run_with_dev_server(run_id, config_dict, task_queue)
```

If no Temporal address is configured, tenfabric auto-starts an embedded development server via `WorkflowEnvironment.start_local()`. This is convenient for testing but not recommended for production.

**Best for:** Production workloads, long-running jobs, scenarios where reliability and observability matter.

### Dispatch Logic

From `cli/train.py:68-128`:

```python
is_local = cfg.infra.provider == InfraProvider.LOCAL

if is_local:
    _run_local(cfg)       # → LocalPipeline
else:
    _run_cloud(cfg)       # → check for Temporal

def _run_cloud(cfg):
    if cfg.workflow.temporal_address:
        TemporalPipeline().run(cfg)   # → Temporal workflow
    else:
        CloudPipeline().run(cfg)       # → Direct SkyPilot
```

The decision tree:
1. Is `provider` local? → `LocalPipeline`
2. Is `temporal_address` set? → `TemporalPipeline`
3. Otherwise → `CloudPipeline`

---

## Adding a New Infrastructure Provider

To add a provider (e.g., CoreWeave):

1. Add enum value in `config/schema.py`:
   ```python
   class InfraProvider(str, Enum):
       # ... existing
       COREWEAVE = "coreweave"
   ```

2. Create `infra/coreweave.py` implementing the Protocol:
   ```python
   class CoreWeaveProvider:
       def provision(self, config: TenfabricConfig) -> InfraHandle: ...
       def setup(self, handle: InfraHandle, config: TenfabricConfig) -> None: ...
       def teardown(self, handle: InfraHandle) -> None: ...
       def status(self, handle: InfraHandle) -> str: ...
   ```

3. Add provider mapping in `infra/skypilot.py` (if using SkyPilot) or add dispatch in `core/pipeline.py` (if custom).

4. Add pricing data in `config/defaults.py`.

No inheritance needed — the Protocol ensures type safety without coupling.

---

## Further Reading

- **[Training Pipeline Deep Dive](training-pipeline.md)** — How the 6-step training pipeline works, from config to model
- **[RunPod Guide](skypilot-runpod.md)** — RunPod-specific pricing, setup, and gotchas
- **[Cloud RunPod Example](examples/06-cloud-runpod/guide.md)** — Hands-on walkthrough
