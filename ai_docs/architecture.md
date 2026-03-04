# Architecture — tenfabric

## System Overview

tenfabric bridges the gap between ML training libraries (TRL, Unsloth) and cloud infrastructure (SkyPilot, Temporal). Users write a single `tenfabric.yaml`, and `tfab train` handles: config validation → GPU selection → infrastructure provisioning → dataset loading → model training → artifact export → teardown. The same config works locally on a laptop GPU or on cloud VMs across AWS/GCP/Azure/RunPod/Lambda.

## Data Flow: Config → Trained Model

```
tenfabric.yaml
      │
      ▼
┌──────────────┐     ┌──────────────────┐
│ config/       │     │ cli/train.py     │
│ loader.py     │────▶│ Parse CLI args   │
│ schema.py     │     │ Apply overrides  │
│ (Pydantic)    │     │ Show plan (Rich) │
└──────────────┘     └────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
              provider=local     provider=cloud
                    │                   │
                    ▼                   ▼
          ┌─────────────┐    ┌──────────────────┐
          │ LocalPipeline│    │ TemporalPipeline │
          │ (pipeline.py)│    │ (pipeline.py)    │
          └──────┬──────┘    └────────┬─────────┘
                 │                    │
                 │              ┌─────▼──────────┐
                 │              │ workflows/      │
                 │              │ client.py       │──▶ Temporal Server
                 │              │ training_       │
                 │              │ pipeline.py     │
                 │              │ activities.py   │
                 │              └─────┬───────────┘
                 │                    │
                 │              ┌─────▼──────────┐
                 │              │ infra/          │
                 │              │ skypilot.py     │──▶ Cloud VM
                 │              │ (generates YAML)│     │
                 │              └─────────────────┘     │
                 │                                      │
                 │  ┌───────────────────────────────────┘
                 │  │  Cloud VM runs: tfab train config.yaml --local
                 │  │  (Re-entry pattern — see below)
                 ▼  ▼
          ┌─────────────────┐
          │ Step 1: Validate │ config already validated by Pydantic
          │ Step 2: GPU     │ torch.cuda detection (lazy import)
          │ Step 3: Dataset │ training/data.py → {"text": ...}
          │ Step 4: Model   │ trl_backend.py or unsloth_backend.py
          │ Step 5: Train   │ SFTTrainer / DPOTrainer / GRPOTrainer
          │ Step 6: Export  │ training/export.py → merge/GGUF/Hub
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ ./outputs/      │ Merged model, adapter, GGUF
          │ runs.db         │ Run metadata (SQLite)
          │ HuggingFace Hub │ Optional push
          └─────────────────┘
```

## Module Dependency Graph

```
                    ┌─────────────┐
                    │ config/     │
                    │ schema.py   │ ◀── EVERYTHING depends on this
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
      ┌────────────┐ ┌──────────┐ ┌──────────────┐
      │ config/    │ │ config/  │ │ infra/       │
      │ loader.py  │ │defaults. │ │ base.py      │
      │ (yaml→obj) │ │py (VRAM) │ │ (Protocol)   │
      └─────┬──────┘ └────┬─────┘ └──────┬───────┘
            │              │              │
            │         ┌────┴────┐    ┌────┴────────┐
            │         ▼         ▼    ▼             ▼
            │   ┌──────────┐ ┌────────────┐ ┌──────────┐
            │   │gpu_      │ │skypilot.py │ │local.py  │
            │   │advisor.py│ │(cloud YAML)│ │(detect)  │
            │   └──────────┘ └────────────┘ └──────────┘
            │
     ┌──────┴──────────────────────────────────────┐
     ▼                                             ▼
┌─────────────┐                           ┌──────────────┐
│ cli/        │                           │ core/        │
│ app.py      │───imports commands────▶   │ pipeline.py  │
│ train.py    │                           │ (orchestrate)│
│ init.py     │                           └──────┬───────┘
│ doctor.py   │                                  │
│ cost.py     │                           ┌──────┴───────┐
│ status.py   │──────────────────────────▶│ core/        │
│ models.py   │                           │ run_store.py │
│ examples.py │                           │ (SQLite)     │
└─────────────┘                           └──────────────┘
                                                 ▲
                                                 │
                                          ┌──────┴───────┐
                                          │ training/    │
                                          │ trl_backend  │ ◀── lazy imported
                                          │ unsloth_back │     by pipeline.py
                                          │ data.py      │
                                          │ export.py    │
                                          └──────────────┘
                                                 ▲
                                          ┌──────┴───────┐
                                          │ workflows/   │
                                          │ activities   │ ◀── lazy imported
                                          │ pipeline     │     by TemporalPipeline
                                          │ client.py    │
                                          │ worker.py    │
                                          └──────────────┘
```

## Import Boundary Rules

| From → To | Allowed? | Notes |
|-----------|----------|-------|
| `cli/` → `config/` | Yes | CLI loads config directly |
| `cli/` → `training/` | **No** | Would import torch at CLI startup |
| `cli/` → `core/pipeline` | Yes | But only inside function body (lazy) |
| `cli/` → `core/run_store` | Yes | For `tfab status` |
| `core/` → `training/` | Yes | But only via lazy import inside methods |
| `core/` → `infra/` | No direct | Infra accessed through workflows or inline |
| `config/` → anything else | **No** | Config is a leaf dependency |
| `training/` → `config/` | Yes | Reads config for hyperparams |
| `infra/` → `config/` | Yes | Reads infra config section |
| `workflows/` → everything | Yes | Orchestrates all modules (via activities) |

## Three Execution Modes

### 1. Local Mode (`infra.provider: local`)
```
User machine → tfab train → LocalPipeline.run() → training backend → ./outputs/
```
- No Temporal, no SkyPilot
- Direct in-process execution: validate → detect GPU → load data → train → export
- All steps sequential in the same Python process
- Code path: `cli/train.py` → `core/pipeline.py:LocalPipeline`

### 2. Cloud Mode (`infra.provider: aws/gcp/runpod/...`)
```
User machine → tfab train → TemporalPipeline → Temporal → SkyPilot → Cloud VM
Cloud VM → tfab train config.yaml --local → LocalPipeline → training → export
```
- Temporal workflow provides durability (retry on spot preemption)
- SkyPilot provisions GPU VM and uploads config
- Training runs on the VM via re-entry (see below)
- Code path: `cli/train.py` → `core/pipeline.py:TemporalPipeline` → `workflows/client.py`

### 3. Production Mode (Temporal with persistent server)
```
Same as Cloud Mode but with a production Temporal server (not auto-started dev server)
```
- Set `workflow.temporal_address` to your Temporal cluster
- Worker runs as a separate process: `tenfabric.workflows.worker:start_worker`
- Provides: workflow history, observability, cross-run orchestration

## The Re-entry Pattern

This is a critical design detail for understanding cloud execution:

```
┌──────────────────────────────┐
│ User's machine               │
│                              │
│ tfab train                   │
│   → SkyPilot generates YAML │
│   → sky launch cluster.yaml  │
│   → uploads tenfabric.yaml  │
└──────────────┬───────────────┘
               │ provisions VM
               ▼
┌──────────────────────────────┐
│ Cloud VM                     │
│                              │
│ SkyPilot run command:        │
│   tfab train /tmp/config.yaml --local  │
│                              │
│ The --local flag forces      │
│ LocalPipeline execution      │
│ (no recursive cloud launch)  │
└──────────────────────────────┘
```

The SkyPilot YAML's `run` field is: `tfab train /tmp/tenfabric-config.yaml --local`

This means tenfabric is installed on the VM (`pip install 'tenfabric[training]'` in setup), and the same CLI runs training locally on the GPU VM. The `--local` flag prevents infinite cloud provisioning loops.

See `infra/skypilot.py:_generate_sky_yaml()` line 155.

## State and Storage

| Location | Contents | Created By |
|----------|----------|------------|
| `~/.tenfabric/runs.db` | SQLite with run metadata (id, project, model, status, timestamps) | `core/run_store.py` |
| `~/.tenfabric/skypilot/` | Generated SkyPilot YAML files per project | `infra/skypilot.py` |
| `./outputs/` (configurable) | Trained model, adapter weights, merged model, GGUF | `training/export.py` |
| `./outputs/merged/` | Merged base+adapter weights | `training/export.py` |
| `./outputs/gguf/` | GGUF quantized model for llama.cpp | `training/export.py` |
| `./tenfabric.yaml` | User's training config | `cli/init.py` or manual |
