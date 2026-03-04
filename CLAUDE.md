# tenfabric — AI Agent Context

## Quick Commands
```bash
uv sync                                  # Install deps
uv run pytest tests/ -v                  # Full test suite (119 tests, <1s, no GPU)
uv run pytest tests/test_config.py -v    # Run one module
uv run ruff check src/ tests/            # Lint
uv run ruff format src/ tests/           # Format
uv run mypy src/tenfabric/               # Type check
uv run tfab --help                       # CLI entry point
```

## Project Overview
tenfabric is a CLI tool that provisions infrastructure, trains fine-tuned LLMs, and exports artifacts — all from a single YAML config. Tech stack: Typer (CLI), Pydantic v2 (config), Rich (output), TRL/Unsloth (training), SkyPilot (cloud), Temporal (durable workflows). Entry point: `tfab` → `src/tenfabric/cli/app.py`.

## Architecture Map
```
src/tenfabric/
├── __init__.py                  # __version__ only
├── cli/                         # CLI commands (Typer). NEVER imports torch.
│   ├── app.py                   # Typer app + command registration
│   ├── train.py                 # tfab train — orchestrates local/cloud
│   ├── init.py                  # tfab init — YAML templates (5 built-in)
│   ├── doctor.py                # tfab doctor — env diagnostics
│   ├── cost.py                  # tfab cost — cloud cost estimation
│   ├── status.py                # tfab status — run history from SQLite
│   ├── models.py                # tfab models — model browser
│   └── examples.py              # tfab examples — example viewer/copier
├── config/                      # YAML schema + loading. NO heavy deps.
│   ├── schema.py                # ⚠ SACRED — Pydantic models, enums, validators
│   ├── loader.py                # YAML → TenfabricConfig, auto-discovery
│   └── defaults.py              # VRAM tables, GPU specs, cloud pricing
├── core/                        # Orchestration layer
│   ├── pipeline.py              # LocalPipeline / TemporalPipeline
│   └── run_store.py             # SQLite run history (~/.tenfabric/runs.db)
├── infra/                       # Infrastructure providers
│   ├── base.py                  # InfraHandle dataclass + InfraProvider Protocol
│   ├── local.py                 # LocalProvider — detect local GPU
│   ├── skypilot.py              # SkyPilotProvider — cloud GPU via SkyPilot
│   └── gpu_advisor.py           # GPU feasibility + cost recommendations
├── training/                    # Training backends. ALWAYS lazy-imported.
│   ├── trl_backend.py           # TRL: prepare_model() + train() (SFT/DPO/GRPO)
│   ├── unsloth_backend.py       # Unsloth: prepare_model() + train() (SFT/DPO)
│   ├── data.py                  # Dataset formatters (alpaca, sharegpt, custom)
│   └── export.py                # Merge adapters, GGUF, Hub push
└── workflows/                   # Temporal durable workflows
    ├── training_pipeline.py     # 7-step workflow (validate→provision→train→teardown)
    ├── activities.py            # Temporal activities wrapping training/infra
    ├── client.py                # Start workflows, auto-start dev server
    └── worker.py                # Temporal worker entry point
```

## Critical Invariants
1. **schema.py is sacred** — All 119 tests depend on it. Change schema → run ALL tests.
2. **No GPU in tests** — Tests run on CPU in <1s. Mock `torch.cuda` if needed.
3. **No network in tests** — No HuggingFace downloads, no Temporal server.
4. **Lazy imports for heavy deps** — `torch`, `transformers`, `trl`, `unsloth`, `datasets`, `sky` are imported inside functions, never at module top level.
5. **cli/ never imports torch** — CLI must start instantly. Training deps are lazy-loaded via `core/pipeline.py`.
6. **Dataset formatters return `{"text": ...}`** — All formatters (`_format_alpaca`, `_format_sharegpt`) must produce a dict with a `"text"` key.
7. **InfraProvider is a Protocol, not ABC** — New providers implement `provision/setup/teardown/status` without inheriting.
8. **Backend dispatch via string matching** — `config.training.backend.value` ("trl"/"unsloth") selects the backend in `pipeline.py`.
9. **Cloud re-entry pattern** — Cloud VMs run `tfab train config.yaml --local`. The `--local` flag forces local execution on the provisioned VM.

## Key Conventions
- **Naming**: snake_case everywhere, Pydantic models are PascalCase
- **Typing**: `from __future__ import annotations` in every file, `X | None` not `Optional[X]`
- **Output**: All user-facing output via `rich.console.Console` — Panel for plans, Table for listings, Spinner for progress
- **Imports**: stdlib → third-party → tenfabric (ruff I rule enforced)
- **Config**: All config flows through `TenfabricConfig` Pydantic model — no loose dicts
- **Errors**: Pydantic `ValidationError` → pretty-printed via Rich → `SystemExit(1)`. `ImportError` for missing deps → `RuntimeError` with install hint.
- **Line length**: 100 chars (ruff)

## Common Tasks

### Add a CLI Command
1. Create `src/tenfabric/cli/mycommand.py` — function with `typer.Argument`/`typer.Option` params
2. Register in `cli/app.py`: `app.command(name="mycommand", help="...")(mycommand_cmd)`
3. Add tests in `tests/test_cli.py` using `CliRunner().invoke(app, [...])`

### Add a Training Backend
1. Add enum value to `TrainingBackend` in `config/schema.py`
2. Create `src/tenfabric/training/new_backend.py` — implement `prepare_model(config) → (model, tokenizer)` and `train(config, model, tokenizer, dataset)`
3. Add dispatch branch in `core/pipeline.py` `_prepare_model()` and `_train()`
4. Add lazy import in `workflows/activities.py` `train_model()`
5. Add optional dep group in `pyproject.toml`

### Add a Dataset Format
1. Add enum value to `DatasetFormat` in `config/schema.py`
2. Add `_format_xxx(example) → {"text": ...}` function in `training/data.py`
3. Add branch in `load_and_format_dataset()` in `training/data.py`
4. Add tests in `tests/test_data.py`

### Add an Infrastructure Provider
1. Add enum value to `InfraProvider` in `config/schema.py`
2. Create `src/tenfabric/infra/new_provider.py` — implement the `InfraProvider` Protocol (provision/setup/teardown/status)
3. Add dispatch in `core/pipeline.py` or `workflows/activities.py`
4. Add tests in `tests/test_infra.py`

## What NOT To Do
1. **Don't import torch/transformers/trl at module level** — Breaks CLI startup
2. **Don't modify schema.py without running full test suite** — Ripple effects everywhere
3. **Don't add GPU-dependent assertions in tests** — Tests must pass on CPU-only CI
4. **Don't bypass Pydantic validation** — Always construct `TenfabricConfig`, never pass raw dicts through the system
5. **Don't hardcode cloud credentials** — SkyPilot handles auth via `sky check`
6. **Don't add deps to core install** — Heavy deps go in optional groups: `[training]`, `[unsloth]`, `[cloud]`

## Deeper Reference
See `ai_docs/` for detailed guides:
- `architecture.md` — Data flow diagrams, module dependencies, execution modes
- `module-reference.md` — Every module's purpose, exports, dependencies, invariants
- `patterns.md` — 8 code patterns with examples and anti-patterns
- `extending.md` — Step-by-step guides for adding features
- `testing.md` — Test philosophy, mocking patterns, coverage gaps
