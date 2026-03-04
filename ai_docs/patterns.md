# Code Patterns — tenfabric

8 patterns used throughout the codebase. Each includes: what, why, example, anti-pattern.

---

## 1. Pydantic Config Pattern

**What**: Nested Pydantic BaseModel classes with typed fields, defaults, constraints, and cross-field validators.

**Why**: Single source of truth for config. Validates at parse time. Provides IDE autocomplete and documentation via Field descriptions.

**Example** (from `config/schema.py`):
```python
class TrainingConfig(BaseModel):
    backend: TrainingBackend = TrainingBackend.TRL
    epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    max_seq_length: int = Field(default=2048, ge=128)

class TenfabricConfig(BaseModel):
    project: str = Field(description="Project name for this training run.")
    model: ModelConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @model_validator(mode="after")
    def validate_quantization_method(self) -> TenfabricConfig:
        if self.model.method == FinetuneMethod.FULL and self.model.quantization != Quantization.NONE:
            raise ValueError("Full fine-tuning does not support quantization.")
        return self
```

**Anti-pattern**: Passing raw dicts through the system. Always construct `TenfabricConfig(**raw)` at the boundary.

---

## 2. Protocol Provider Pattern

**What**: Use `typing.Protocol` for infrastructure providers instead of ABC inheritance.

**Why**: Structural typing — any class with the right methods satisfies the protocol. No inheritance coupling. Easy to add new providers.

**Example** (from `infra/base.py`):
```python
class InfraProvider(Protocol):
    def provision(self, config: TenfabricConfig) -> InfraHandle: ...
    def setup(self, handle: InfraHandle, config: TenfabricConfig) -> None: ...
    def teardown(self, handle: InfraHandle) -> None: ...
    def status(self, handle: InfraHandle) -> str: ...
```

Implementations just define these methods:
```python
class LocalProvider:
    def provision(self, config: TenfabricConfig) -> InfraHandle:
        gpu_name, gpu_count = _detect_local_gpu()
        return InfraHandle(provider="local", instance_id="localhost", ...)
```

**Anti-pattern**: Using `class MyProvider(InfraProvider)` or ABC. The Protocol is not inherited.

---

## 3. Backend Dispatch Pattern

**What**: Select training backend via string comparison on `config.training.backend.value`, with lazy imports of the chosen backend.

**Why**: Only the selected backend's dependencies are imported. Users without Unsloth can still use TRL.

**Example** (from `core/pipeline.py`):
```python
def _prepare_model(self, config: TenfabricConfig) -> None:
    backend = config.training.backend.value
    if backend == "trl":
        self._prepare_trl(config)
    else:
        self._prepare_unsloth(config)

def _prepare_trl(self, config: TenfabricConfig) -> None:
    from tenfabric.training.trl_backend import prepare_model
    self._model, self._tokenizer = prepare_model(config)
```

Training method dispatch within a backend follows the same pattern:
```python
def train(config, model, tokenizer, dataset):
    method = config.training.method
    if method == TrainingMethod.SFT:
        _train_sft(config, model, tokenizer, dataset)
    elif method == TrainingMethod.DPO:
        _train_dpo(config, model, tokenizer, dataset)
```

**Anti-pattern**: Importing all backends at module level. Or using a registry dict — the if/else is explicit and greppable.

---

## 4. Lazy Import Pattern

**What**: Import heavy dependencies (`torch`, `transformers`, `trl`, `unsloth`, `datasets`, `sky`) inside functions, not at module top level.

**Why**: CLI must start instantly (<100ms). Training deps take seconds to import. Users who only run `tfab init` shouldn't wait for torch.

**Example** (from `training/trl_backend.py`):
```python
def prepare_model(config: TenfabricConfig) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import torch
    # ... use them
```

**Example** (from `core/pipeline.py`):
```python
def _detect_gpu(self, config: TenfabricConfig) -> None:
    try:
        import torch
        if torch.cuda.is_available():
            # ...
    except ImportError:
        raise RuntimeError("PyTorch not installed. Install: pip install 'tenfabric[training]'")
```

**Anti-pattern**: `import torch` at the top of a module that's imported by `cli/`. This breaks `tfab --help` for users without torch.

---

## 5. CLI Command Pattern

**What**: Each command is a standalone function with `typer.Argument`/`typer.Option` parameters, registered in `app.py`.

**Why**: Modular — each command in its own file. Registration is explicit. Easy to test with `CliRunner`.

**Example** (from `cli/train.py` + `cli/app.py`):
```python
# train.py
def train_cmd(
    config: Optional[str] = typer.Argument(None, help="Path to config file."),
    local: bool = typer.Option(False, "--local", help="Force local execution."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show plan without running."),
) -> None:
    cfg = load_config(Path(config) if config else None)
    # ...

# app.py
app.command(name="train", help="Train a model from config.")(train_cmd)
```

**Anti-pattern**: Using `@app.command()` decorator directly in `app.py` with inline logic. Commands belong in separate files.

---

## 6. Test Helper Pattern

**What**: Use a `_make_config(**overrides)` helper to construct `TenfabricConfig` with sensible defaults, overriding only what the test needs.

**Why**: Reduces boilerplate. Tests focus on what's being tested. Changes to unrelated defaults don't break tests.

**Example** (typical test pattern):
```python
def _make_config(**overrides):
    base = {
        "project": "test",
        "model": {"base": "test/model", "method": "lora", "quantization": "4bit"},
        "dataset": {"source": "test/dataset", "format": "alpaca"},
    }
    base.update(overrides)
    return TenfabricConfig(**base)

def test_qlora_requires_quantization():
    with pytest.raises(ValidationError, match="QLoRA requires quantization"):
        _make_config(model={"base": "x", "method": "qlora", "quantization": "none"})
```

**Anti-pattern**: Copy-pasting full config dicts into every test function.

---

## 7. Error Handling Pattern

**What**: Three error paths depending on source:

1. **Config validation** → Pydantic `ValidationError` → pretty-print with Rich → `SystemExit(1)`
2. **Missing dependencies** → `ImportError` → `RuntimeError` with install instructions
3. **Runtime failures** → Catch, log to RunStore, print with Rich, re-raise

**Example — Config error** (from `config/loader.py`):
```python
try:
    config = TenfabricConfig(**raw)
except ValidationError as e:
    _print_validation_errors(e, config_path)  # Rich formatted
    raise SystemExit(1) from e
```

**Example — Missing dep** (from `training/unsloth_backend.py`):
```python
try:
    from unsloth import FastLanguageModel
except ImportError:
    raise RuntimeError(
        "Unsloth not installed. Install it with:\n"
        "  pip install 'tenfabric[unsloth]'\n"
        "Or switch to TRL backend: training.backend: trl"
    )
```

**Example — Runtime failure** (from `core/pipeline.py`):
```python
except Exception as e:
    self.store.update(run_id, status="failed", error=str(e))
    console.print(f"\n[bold red]Training failed:[/] {e}")
    raise
```

**Anti-pattern**: Bare `except:` clauses. Or swallowing errors silently. Or printing raw tracebacks to users.

---

## 8. Rich Output Pattern

**What**: All user-facing output goes through `rich.console.Console` using:
- `Panel` — For framed information (execution plan, init success, doctor results)
- `Table` — For tabular data (doctor checks, model browser, status)
- `Spinner` with `Live` — For progress during pipeline steps
- Color conventions: green=success, red=error, yellow=warning, cyan=info, dim=secondary

**Example — Panel** (from `cli/train.py`):
```python
console.print(
    Panel(
        "\n".join(plan_lines),
        title="[bold blue]Execution Plan[/]",
        border_style="blue",
    )
)
```

**Example — Spinner** (from `core/pipeline.py`):
```python
with Live(
    Spinner("dots", text=Text(f"  Step {i}/{len(steps)}: {name}...")),
    console=console,
    transient=True,
):
    step_fn(config)
console.print(f"  [green]✓[/] Step {i}/{len(steps)}: {name}")
```

**Anti-pattern**: Using `print()` or `logging.info()` for user-facing output. Or mixing Rich markup styles inconsistently.
