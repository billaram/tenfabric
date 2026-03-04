# Tenfabric Test Guide

A mental model for the test suite — what each test does, why it exists, and which tests break when you change each module.

## How to Run

```bash
# All tests
uv run pytest tests/ -v

# Single module
uv run pytest tests/test_config.py -v

# Single test class
uv run pytest tests/test_cli.py::TestInit -v

# By keyword
uv run pytest tests/ -k "skypilot" -v
```

---

## Architecture ↔ Test Map

```
tenfabric/
├── config/           ← test_config.py, test_config_edge_cases.py, test_defaults.py
│   ├── schema.py         (Pydantic models — MOST tests depend on this)
│   ├── loader.py         (YAML → TenfabricConfig)
│   └── defaults.py       (VRAM tables, GPU advisor data)
│
├── cli/              ← test_cli.py
│   ├── app.py            (Typer app wiring)
│   ├── init.py           (template scaffolding)
│   ├── train.py          (dry-run, provider override)
│   ├── doctor.py         (environment checks)
│   ├── cost.py           (cost estimation)
│   ├── models.py         (model browser)
│   ├── examples.py       (example viewer/copier)
│   └── status.py         ⚠ NO TESTS
│
├── training/         ← test_data.py
│   ├── data.py           (dataset formatters)
│   ├── trl_backend.py    ⚠ NO TESTS (needs GPU/mocks)
│   ├── unsloth_backend.py⚠ NO TESTS (needs GPU/mocks)
│   └── export.py         ⚠ NO TESTS (needs GPU/mocks)
│
├── infra/            ← test_infra.py, test_skypilot_config.py, test_gpu_advisor.py
│   ├── base.py           (InfraHandle dataclass, protocol)
│   ├── local.py          (local GPU provider)
│   ├── skypilot.py       (YAML generation tested, cloud calls not)
│   └── gpu_advisor.py    (advice engine)
│
├── core/             ← test_run_store.py
│   ├── run_store.py      (SQLite CRUD)
│   └── pipeline.py       ⚠ NO TESTS (orchestrates everything)
│
└── workflows/        ⚠ NO TESTS (all require Temporal runtime)
    ├── training_pipeline.py
    ├── activities.py
    ├── worker.py
    └── client.py
```

---

## Test-by-Test Reference

### test_config.py — 7 tests
**Purpose:** Core config validation. If these break, the entire CLI is broken.

| Test | Why It Exists |
|------|---------------|
| `test_minimal_config` | Proves defaults work — user only needs `project`, `model.base`, `dataset.source` |
| `test_full_config` | Proves every field is parseable when explicitly set |
| `test_full_finetune_rejects_quantization` | Guards the cross-field rule: `method: full` + `quantization: 4bit` is invalid |
| `test_qlora_requires_quantization` | Guards the cross-field rule: `method: qlora` + `quantization: none` is invalid |
| `test_load_config_from_yaml` | Proves YAML → Pydantic roundtrip works |
| `test_load_config_missing_required` | Proves helpful error when user forgets `model` or `dataset` |
| `test_load_config_file_not_found` | Proves clean error instead of traceback |

### test_config_edge_cases.py — 22 tests
**Purpose:** Exhaustive boundary testing for every config field.

| Test Group | Why It Exists |
|------------|---------------|
| `TestModelConfig` (2) | Every `FinetuneMethod` and `Quantization` enum value creates a valid config |
| `TestTrainingConfig` (6) | Field constraints: epochs > 0, batch_size > 0, lr > 0, seq_length >= 128, max_steps default |
| `TestLoraConfig` (4) | Defaults are sane, explicit modules work, r > 0, dropout ∈ [0, 1] |
| `TestInfraConfig` (4) | All 7 providers valid, budget_max >= 0, disk_size >= 10 |
| `TestDatasetConfig` (3) | All 3 formats valid, max_samples > 0, custom template passthrough |
| `TestCrossFieldValidation` (2) | Valid combos: full+none, lora+any |
| `TestConfigAutoDiscovery` (6) | Finds `.yaml`, finds `.yml`, prefers `.yaml`, error when missing, explicit path works |
| `TestLoadConfigYaml` (3) | Empty YAML error, extra fields tolerated, full roundtrip preserves all values |

### test_cli.py — 26 tests
**Purpose:** Every CLI command works end-to-end through Typer.

| Test Group | Tests | Why It Exists |
|------------|-------|---------------|
| `TestVersion` | 2 | `--version` and `-v` flags work |
| `TestInit` | 10 | All 5 templates scaffold valid configs, unknown template errors, no-overwrite guard, force-overwrite, all templates pass validation |
| `TestExamples` | 4 | List/view/copy examples, unknown example errors |
| `TestModels` | 4 | List all, filter by size, filter by family, no-match message |
| `TestDoctor` | 1 | Runs without crashing, shows Python version |
| `TestTrain` | 4 | No-config error, missing file error, dry-run shows plan, invalid provider error |
| `TestCost` | 2 | Cost estimate for known model, error for unknown model |

### test_defaults.py — 8 tests
**Purpose:** The GPU recommendation engine's data tables are correct.

| Test | Why It Exists |
|------|---------------|
| `test_guess_model_size_known` | Model ID → param count mapping works for Llama, Qwen, Phi |
| `test_guess_model_size_unknown` | Returns `None` (not crash) for unrecognized models |
| `test_estimate_vram_lora_4bit` | 1B + LoRA 4bit → 2-5 GB (sanity range) |
| `test_estimate_vram_full_finetune` | 7B + full → 40-70 GB (sanity range) |
| `test_recommend_gpu` | All recommended GPUs have VRAM >= requirement + 10% headroom |
| `test_recommend_gpu_tiny` | Even 2GB requirement returns at least one GPU |
| `test_cheapest_cloud_option` | Returns (provider, cost) with cost > 0 for known GPUs |
| `test_cheapest_cloud_option_unknown` | Returns `None` for unknown GPUs |

### test_gpu_advisor.py — 7 tests
**Purpose:** The advice engine integrates defaults + local detection correctly.

| Test | Why It Exists |
|------|---------------|
| `test_known_small_model` | 1B model → low VRAM, multiple GPUs recommended |
| `test_known_large_model` | 70B model → high VRAM (>20GB) |
| `test_unknown_model_defaults_to_7b` | Unknown model produces warning, doesn't crash |
| `test_local_gpu_sufficient` | RTX 4090 (24GB) is feasible for 1B model |
| `test_local_gpu_insufficient` | RTX 3060 (12GB) is infeasible for 70B model, warning issued |
| `test_no_local_gpu` | No GPU → local_feasible = False |
| `test_cheapest_cloud_populated` | Cheapest cloud option is always returned for valid configs |

### test_data.py — 9 tests
**Purpose:** Dataset formatters produce correct training text.

| Test Group | Tests | Why It Exists |
|------------|-------|---------------|
| `TestFormatAlpaca` | 4 | With/without input field, missing fields, empty example all produce `{"text": ...}` |
| `TestFormatShareGPT` | 5 | Multi-turn conversations, system messages, both key styles (`from`/`role`), empty/missing gracefully handled |

### test_infra.py — 8 tests
**Purpose:** Local provider works, InfraHandle dataclass is correct.

| Test Group | Tests | Why It Exists |
|------------|-------|---------------|
| `TestLocalProvider` | 6 | Provision with GPU, without GPU, Apple Silicon. Setup/teardown are no-ops. Status returns "ready". |
| `TestInfraHandle` | 2 | Default values are sensible, full construction works |

### test_skypilot_config.py — 17 tests
**Purpose:** SkyPilot YAML generation is correct (no cloud calls).

| Test Group | Tests | Why It Exists |
|------------|-------|---------------|
| `TestGenerateSkyYaml` | 14 | Structure valid, spot/no-spot, explicit GPU, all cloud providers, disk size, setup includes tenfabric, unsloth added when needed, run command uses `--local`, config mounted, custom envs pass through, region handling |
| `TestAutoSelectGpu` | 3 | Small model → cheap GPU, large model → big GPU, unknown model → safe default (A10G) |

### test_run_store.py — 5 tests
**Purpose:** SQLite run tracking CRUD.

| Test | Why It Exists |
|------|---------------|
| `test_create_and_get` | Create a run, retrieve it, status is "pending" |
| `test_update_status` | Status → "completed" sets finished_at timestamp |
| `test_update_error` | Failed run stores error message |
| `test_list_recent` | Pagination (limit) works |
| `test_get_nonexistent` | Returns None, not crash |

---

## Change Impact Matrix

**When you change a module, these tests MUST pass:**

| If you change... | Run these tests | Why |
|-------------------|----------------|-----|
| `config/schema.py` | **ALL TESTS** | Every module depends on TenfabricConfig |
| `config/loader.py` | `test_config`, `test_config_edge_cases`, `test_cli` | Config loading feeds everything |
| `config/defaults.py` | `test_defaults`, `test_gpu_advisor`, `test_skypilot_config`, `test_cli::TestCost`, `test_cli::TestTrain::test_train_dry_run` | VRAM tables, GPU data, cost estimates |
| `cli/app.py` | `test_cli` (all) | App wiring affects every command |
| `cli/init.py` | `test_cli::TestInit` | Template content and scaffolding |
| `cli/train.py` | `test_cli::TestTrain` | Dry-run, provider override, config loading |
| `cli/doctor.py` | `test_cli::TestDoctor` | Environment checks |
| `cli/cost.py` | `test_cli::TestCost` | Cost estimation logic |
| `cli/models.py` | `test_cli::TestModels` | Model browser data and filtering |
| `cli/examples.py` | `test_cli::TestExamples` | Example listing and copying |
| `cli/status.py` | ⚠ No tests yet | — |
| `training/data.py` | `test_data` | Dataset formatters |
| `training/trl_backend.py` | ⚠ No tests yet | — |
| `training/unsloth_backend.py` | ⚠ No tests yet | — |
| `training/export.py` | ⚠ No tests yet | — |
| `infra/base.py` | `test_infra` | InfraHandle, provider protocol |
| `infra/local.py` | `test_infra::TestLocalProvider` | Local GPU detection and provisioning |
| `infra/skypilot.py` | `test_skypilot_config` | YAML generation and GPU auto-selection |
| `infra/gpu_advisor.py` | `test_gpu_advisor` | Advice engine, feasibility checks |
| `core/run_store.py` | `test_run_store` | SQLite run history |
| `core/pipeline.py` | ⚠ No tests yet | — |
| `workflows/*` | ⚠ No tests yet | — |

### Quick Commands by Change Area

```bash
# Changed config schema or loader
uv run pytest tests/ -v                           # run everything

# Changed a specific CLI command
uv run pytest tests/test_cli.py::TestInit -v      # just init
uv run pytest tests/test_cli.py::TestTrain -v     # just train
uv run pytest tests/test_cli.py::TestCost -v      # just cost

# Changed GPU/VRAM tables or advisor
uv run pytest tests/test_defaults.py tests/test_gpu_advisor.py tests/test_skypilot_config.py -v

# Changed dataset formatting
uv run pytest tests/test_data.py -v

# Changed infra providers
uv run pytest tests/test_infra.py tests/test_skypilot_config.py -v

# Changed run store
uv run pytest tests/test_run_store.py -v
```

---

## Dependency Graph

```
config/schema.py          ← FOUNDATION: almost everything imports this
    ↑
config/loader.py          ← reads YAML, returns TenfabricConfig
    ↑
config/defaults.py        ← VRAM tables, model sizes, GPU specs
    ↑           ↑
    |     infra/gpu_advisor.py  ← combines defaults + local detection
    |           ↑
    |     infra/skypilot.py     ← generates SkyPilot YAML, auto-selects GPU
    |           ↑
    |     infra/local.py        ← local GPU detection
    |     infra/base.py         ← InfraHandle dataclass
    |           ↑
cli/train.py ─────────────────→ core/pipeline.py
cli/cost.py  ─────────────────→ config/defaults.py
cli/init.py  ─────────────────→ (templates, standalone)
cli/doctor.py ────────────────→ (system checks, standalone)
cli/models.py ────────────────→ (static data, standalone)
cli/examples.py ──────────────→ cli/init.py (shares TEMPLATES)
cli/status.py ────────────────→ core/run_store.py
    ↓
core/pipeline.py ─────────────→ training/trl_backend.py
                  ─────────────→ training/unsloth_backend.py
                  ─────────────→ training/export.py
                  ─────────────→ training/data.py
                  ─────────────→ core/run_store.py
                  ─────────────→ workflows/client.py (cloud mode)
    ↓
workflows/client.py ──────────→ workflows/training_pipeline.py
                    ──────────→ workflows/worker.py
workflows/worker.py ──────────→ workflows/activities.py
workflows/activities.py ──────→ training/*, infra/*
```

---

## Coverage Gaps & What's Needed

| Module | Gap | Why | What Would Tests Look Like |
|--------|-----|-----|---------------------------|
| `cli/status.py` | No tests | Renders run data from store | Mock RunStore, assert table output |
| `training/trl_backend.py` | No tests | Requires PyTorch + GPU or heavy mocking | Mock `transformers`, `peft`, `trl` imports; test config → trainer args mapping |
| `training/unsloth_backend.py` | No tests | Requires Unsloth + GPU | Mock `unsloth.FastLanguageModel`; test config → Unsloth args mapping |
| `training/export.py` | No tests | Requires trained model object | Mock model.merge_and_unload, model.push_to_hub; test branching logic |
| `core/pipeline.py` | No tests | Orchestrates everything | Mock all training/infra imports; test step sequence and error handling |
| `workflows/*` | No tests | Requires Temporal server | Use `temporalio.testing.WorkflowEnvironment`; test workflow step sequence and retry behavior |

### Priority for Adding Tests

1. **`cli/status.py`** — Easy, just mock RunStore
2. **`training/export.py`** — Medium, mock model objects, test branching (merge/gguf/hub)
3. **`core/pipeline.py`** — Medium, mock all backends, test the orchestration sequence
4. **`training/trl_backend.py`** — Hard, heavy mocking of HuggingFace ecosystem
5. **`workflows/*`** — Hard, needs Temporal test environment

---

## Test Design Principles

1. **No GPU required.** All tests run on CPU-only CI. GPU-dependent code is mocked.
2. **No network calls.** No HuggingFace downloads, no cloud API calls. SkyPilot tests only verify YAML generation.
3. **No Temporal server.** Workflow tests (when added) should use `temporalio.testing`.
4. **Fast.** Full suite runs in < 1 second. Keep it that way.
5. **`config/schema.py` is sacred.** Changes there ripple everywhere. Always run the full suite.
