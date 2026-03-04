# Module Reference — tenfabric

Every module documented with: purpose, key exports, dependencies, dependents, invariants.

---

## config/schema.py — FOUNDATION

**Purpose**: Pydantic v2 models defining the entire `tenfabric.yaml` schema. Every other module imports from here.

**Key Exports**:
- `TenfabricConfig` — Root config model (project, version, model, dataset, training, lora, infra, workflow, output)
- Enums: `TrainingMethod` (sft/dpo/grpo/ppo/kto/orpo), `FinetuneMethod` (lora/qlora/full), `Quantization` (none/4bit/8bit), `DatasetFormat` (alpaca/sharegpt/custom), `InfraProvider` (auto/local/aws/gcp/azure/runpod/lambda), `TrainingBackend` (trl/unsloth)
- Section models: `ModelConfig`, `DatasetConfig`, `TrainingConfig`, `LoraConfig`, `InfraConfig`, `WorkflowConfig`, `OutputConfig`, `SkyPilotPassthrough`, `RetryPolicy`

**Validators**:
- `validate_quantization_method` — Full fine-tune rejects quantization != none
- `validate_qlora_quantization` — QLoRA requires quantization != none

**Dependencies**: pydantic only
**Depended on by**: Every module in the project
**Invariants**: Changes here require running ALL 119 tests. Never remove enum values without checking all dispatch points.

---

## config/loader.py

**Purpose**: Load and validate YAML config files into `TenfabricConfig`.

**Key Exports**:
- `load_config(path?) → TenfabricConfig` — Main entry point. Auto-discovers `tenfabric.yaml`/`.yml`.
- `find_config(path?) → Path` — Resolve config file path.

**Dependencies**: config/schema.py, pyyaml, rich
**Depended on by**: cli/train.py, cli/cost.py
**Invariants**: Always returns a validated `TenfabricConfig`. Validation errors pretty-printed via Rich then `SystemExit(1)`.

---

## config/defaults.py

**Purpose**: Static tables for VRAM estimation, GPU specs, model sizes, and cloud pricing.

**Key Exports**:
- `VRAM_ESTIMATES` — dict: param_billions → method → VRAM_GB
- `MODEL_SIZES` — dict: model name fragment → param count (0.135 to 72.0)
- `GPU_VRAM` — dict: GPU name → VRAM_GB (T4 through H100)
- `GPU_SPOT_COSTS` — dict: GPU → provider → $/hr
- `guess_model_size(model_id) → float | None`
- `estimate_vram(model_size_b, method, quantization) → float`
- `recommend_gpu(vram_needed) → list[str]`
- `cheapest_cloud_option(gpu) → tuple[str, float] | None`

**Dependencies**: None (pure data + functions)
**Depended on by**: cli/train.py, cli/cost.py, infra/gpu_advisor.py, infra/skypilot.py
**Invariants**: `recommend_gpu` adds 10% headroom. Unknown model sizes extrapolate linearly from closest known.

---

## cli/app.py

**Purpose**: Typer application entry point. Registers all CLI commands.

**Key Exports**:
- `app` — Typer instance (entry point for `tfab` command)

**Dependencies**: typer, rich, tenfabric.__init__ (version), all cli/*.py command modules
**Depended on by**: pyproject.toml entry point (`tfab = tenfabric.cli.app:app`)
**Invariants**: Commands registered via `app.command(name=...)(fn)` pattern, not decorators. This allows commands in separate files.

---

## cli/train.py

**Purpose**: The `tfab train` command — load config, show plan, dispatch to local or cloud pipeline.

**Key Exports**:
- `train_cmd(config?, local?, provider?, dry_run?)` — Main command function

**Key behavior**:
- `--local` flag → forces `InfraProvider.LOCAL`
- `--provider X` → overrides `infra.provider`
- `--dry-run` → show plan and exit
- Local → `LocalPipeline`, cloud → `TemporalPipeline` (both lazy-imported)

**Dependencies**: config/loader, config/defaults, config/schema
**Depended on by**: cli/app.py
**Invariants**: Pipeline imports are inside function bodies (`_run_local`, `_run_cloud`) — never at top level.

---

## cli/init.py

**Purpose**: `tfab init` — scaffold starter YAML configs from built-in templates.

**Key Exports**:
- `init_cmd(template?, output?, force?)` — Main command
- `TEMPLATES` — dict of 5 template strings: quickstart, lora, qlora, dpo, cloud

**Dependencies**: typer, rich, pathlib
**Depended on by**: cli/app.py, cli/examples.py
**Invariants**: Never overwrites existing file without `--force`.

---

## cli/doctor.py

**Purpose**: `tfab doctor` — check Python, CUDA, PyTorch, GPU, and all dependencies.

**Key Exports**:
- `doctor_cmd()` — Runs all environment checks, displays Rich Table

**Dependencies**: typer, rich, sys, shutil, subprocess
**Depended on by**: cli/app.py
**Invariants**: Does not import torch at module level — catches `ImportError` gracefully.

---

## cli/cost.py

**Purpose**: `tfab cost` — estimate cloud training costs from config.

**Key Exports**:
- `cost_cmd(config?)` — Loads config, estimates VRAM, recommends GPUs with pricing

**Dependencies**: config/loader, config/defaults, rich
**Depended on by**: cli/app.py

---

## cli/status.py

**Purpose**: `tfab status` — view training run history from SQLite.

**Key Exports**:
- `status_cmd(run_id?)` — Show recent runs or specific run details

**Dependencies**: core/run_store, rich
**Depended on by**: cli/app.py

---

## cli/models.py

**Purpose**: `tfab models` — browse recommended base models.

**Key Exports**:
- `models_cmd(filter?)` — Display model table with sizes and recommendations

**Dependencies**: rich, config/defaults
**Depended on by**: cli/app.py

---

## cli/examples.py

**Purpose**: `tfab examples` — browse and copy example configurations.

**Key Exports**:
- `examples_cmd(action?)` — List/view/copy example configs from `docs/examples/`

**Dependencies**: cli/init (shares templates), rich, pathlib
**Depended on by**: cli/app.py

---

## core/pipeline.py

**Purpose**: Training pipeline orchestration. Two implementations: local (in-process) and Temporal (durable).

**Key Exports**:
- `LocalPipeline` — Sequential in-process execution. Steps: validate → GPU detect → dataset → prepare model → train → export. Uses Rich Spinner for progress.
- `TemporalPipeline` — Starts a Temporal workflow via `workflows/client.py`.

**Dependencies**: config/schema, core/run_store, rich. Training deps lazy-imported in step methods.
**Depended on by**: cli/train.py (lazy import)
**Invariants**: All training/export imports are inside methods (`_prepare_trl`, `_train`, `_export`). Never at class or module level.

---

## core/run_store.py

**Purpose**: SQLite-backed run history. Tracks run_id, project, model, provider, status, timestamps.

**Key Exports**:
- `RunStore(db_path?)` — Default path: `~/.tenfabric/runs.db`
- `.create(run_id, project, model, provider, status?)` — Insert new run
- `.update(run_id, status?, error?)` — Update status, set finished_at on terminal states
- `.get(run_id) → dict | None` — Retrieve single run
- `.list_recent(limit=10) → list[dict]` — Recent runs ordered by created_at DESC

**Dependencies**: sqlite3, pathlib, datetime
**Depended on by**: core/pipeline.py, cli/status.py
**Invariants**: Auto-creates DB and table on init. Accepts `db_path` for testing with `tmp_path`.

---

## infra/base.py

**Purpose**: Protocol definition for infrastructure providers.

**Key Exports**:
- `InfraHandle` — Dataclass: provider, instance_id, host, port, gpu_name, gpu_count, status, metadata
- `InfraProvider` — Protocol with 4 methods: `provision(config) → InfraHandle`, `setup(handle, config)`, `teardown(handle)`, `status(handle) → str`

**Dependencies**: config/schema (TenfabricConfig)
**Depended on by**: infra/local.py, infra/skypilot.py, workflows/activities.py
**Invariants**: This is a Protocol (structural typing), not ABC. New providers don't inherit — they just implement the methods.

---

## infra/local.py

**Purpose**: Local infrastructure provider — detects the current machine's GPU.

**Key Exports**:
- `LocalProvider` — Implements InfraProvider Protocol. `provision()` detects GPU via torch.cuda. `setup()`/`teardown()` are no-ops.

**Dependencies**: config/schema, infra/base, (torch — lazy in `_detect_local_gpu`)
**Depended on by**: workflows/activities.py (indirectly)
**Invariants**: `provision()` never raises — returns gpu_name=None and gpu_count=0 if torch not installed.

---

## infra/skypilot.py

**Purpose**: SkyPilot cloud provider — generates SkyPilot YAML and launches cloud VMs.

**Key Exports**:
- `SkyPilotProvider` — Implements InfraProvider Protocol. Uses `sky launch` subprocess.
- `_generate_sky_yaml(config) → dict` — Builds SkyPilot task dict from TenfabricConfig
- `_auto_select_gpu(config) → str` — Picks cheapest suitable GPU

**Key behavior**:
- Generated YAML `run` field: `tfab train /tmp/tenfabric-config.yaml --local` (re-entry pattern)
- `setup` section: `pip install 'tenfabric[training]'` (+ unsloth if needed)
- File mounts: uploads `tenfabric.yaml` to `/tmp/tenfabric-config.yaml`
- YAML written to `~/.tenfabric/skypilot/{project}.yaml`

**Dependencies**: config/schema, config/defaults, infra/base, pyyaml, subprocess
**Depended on by**: workflows/activities.py
**Invariants**: Maps tenfabric provider names to SkyPilot cloud names via `PROVIDER_TO_SKY_CLOUD`. Always generates `--local` in run command.

---

## infra/gpu_advisor.py

**Purpose**: GPU feasibility analysis and cost recommendations.

**Key Exports**:
- `GpuAdvice` — Dataclass with model_size, vram_needed, local_gpu, local_feasible, recommended_gpus, cheapest_cloud, warnings
- `advise(config) → GpuAdvice` — Full analysis
- `print_advice(advice)` — Rich Panel output

**Dependencies**: config/defaults (all functions), config/schema, rich
**Depended on by**: cli/cost.py, cli/doctor.py
**Invariants**: Unknown models default to 7B. Local feasibility threshold is 90% of estimated VRAM.

---

## training/trl_backend.py

**Purpose**: TRL-based training — SFT, DPO, GRPO trainers.

**Key Exports**:
- `prepare_model(config) → (model, tokenizer)` — Load model, apply quantization, apply LoRA
- `train(config, model, tokenizer, dataset)` — Dispatch to `_train_sft`/`_train_dpo`/`_train_grpo`

**Dependencies**: config/schema, transformers, peft, trl, torch (ALL lazy-imported inside functions)
**Depended on by**: core/pipeline.py, workflows/activities.py (both lazy)
**Invariants**: `prepare_model` handles all quantization/LoRA setup. `_auto_detect_target_modules` falls back to `["q_proj", "v_proj"]`. Training methods PPO/KTO/ORPO raise `ValueError` (not yet implemented).

---

## training/unsloth_backend.py

**Purpose**: Unsloth-optimized training — 2x faster, 70% less VRAM.

**Key Exports**:
- `prepare_model(config) → (model, tokenizer)` — Uses `FastLanguageModel.from_pretrained`
- `train(config, model, tokenizer, dataset)` — SFT and DPO only

**Dependencies**: config/schema, unsloth, trl (lazy)
**Depended on by**: core/pipeline.py, workflows/activities.py (both lazy)
**Invariants**: Only supports SFT and DPO. Other methods raise `ValueError` directing user to TRL backend. Uses `use_gradient_checkpointing="unsloth"` (string, not bool).

---

## training/data.py

**Purpose**: Dataset loading and formatting for training.

**Key Exports**:
- `load_and_format_dataset(config) → dataset` — Load from HuggingFace, apply formatter
- `_format_alpaca(example) → {"text": ...}` — Instruction/Input/Response format
- `_format_sharegpt(example) → {"text": ...}` — Multi-turn Human/Assistant format

**Dependencies**: config/schema, datasets (lazy)
**Depended on by**: core/pipeline.py (inline), workflows/activities.py
**Invariants**: ALL formatters must return `{"text": text_string}`. Custom format uses `text_column` rename or passthrough.

---

## training/export.py

**Purpose**: Post-training model export — merge adapters, GGUF conversion, Hub push.

**Key Exports**:
- `export_model(config, model, tokenizer)` — Orchestrates all export steps

**Export steps** (each conditional):
1. Merge LoRA adapter into base (tries Unsloth first, falls back to PEFT)
2. Export GGUF via Unsloth (requires unsloth extra)
3. Push to HuggingFace Hub

**Dependencies**: config/schema, rich, unsloth (optional), peft (optional)
**Depended on by**: core/pipeline.py
**Invariants**: Merge only happens for LoRA/QLoRA methods. GGUF requires unsloth. Hub push requires non-empty `hub_repo`.

---

## workflows/training_pipeline.py

**Purpose**: Temporal workflow definition — 7-step durable training pipeline.

**Key Exports**:
- `TrainingPipelineWorkflow` — `@workflow.defn` class with `run(run_id, config_dict)` method

**Steps**:
1. validate_config (30s timeout)
2. provision_infra (10min, retries per config)
3. setup_environment (15min, 2 retries)
4. prepare_dataset (30min, 2 retries)
5. train_model (24h, 5min heartbeat, retries per config)
6. export_model (30min, 2 retries)
7. teardown_infra (5min, 3 retries, in finally block)

**Dependencies**: temporalio, config/schema (via `workflow.unsafe.imports_passed_through()`)
**Depended on by**: workflows/client.py, workflows/worker.py
**Invariants**: `teardown_infra` is in a `finally` block — always runs even on failure.

---

## workflows/activities.py

**Purpose**: Temporal activity implementations wrapping training and infra operations.

**Key Exports**:
- `validate_config(config_dict) → dict`
- `provision_infra(config_dict) → dict`
- `setup_environment(infra_handle, config_dict) → dict`
- `prepare_dataset(infra_handle, config_dict) → dict`
- `train_model(infra_handle, config_dict) → dict`
- `export_model(infra_handle, config_dict, train_result) → dict`
- `teardown_infra(infra_handle) → dict`

**Dependencies**: temporalio, config/schema, training/* (lazy), infra/* (lazy)
**Depended on by**: workflows/worker.py (registered as activities)
**Invariants**: All activities accept/return plain dicts (Temporal serialization). Config reconstructed from dict inside each activity.

---

## workflows/client.py

**Purpose**: Start training workflows on Temporal and manage the dev server.

**Key Exports**:
- `start_training_workflow(run_id, config, temporal_address?) → dict` — Start workflow and wait for result
- `_ensure_dev_server() → str` — Auto-start embedded Temporal dev server if none running

**Dependencies**: temporalio, subprocess, workflows/training_pipeline
**Depended on by**: core/pipeline.py (TemporalPipeline)
**Invariants**: Starts a background worker process. Worker is terminated in `finally` block. Dev server on `localhost:7233`.

---

## workflows/worker.py

**Purpose**: Temporal worker process — registers workflows and activities.

**Key Exports**:
- `run_worker(temporal_address?, task_queue?) → None` — async worker loop
- `start_worker(temporal_address?, task_queue?) → None` — sync entry point (`asyncio.run`)

**Dependencies**: temporalio, workflows/activities (all 7), workflows/training_pipeline
**Depended on by**: workflows/client.py (spawned as subprocess)
**Invariants**: Default task queue: `"tenfabric-training"`. Registers all 7 activities and TrainingPipelineWorkflow.
