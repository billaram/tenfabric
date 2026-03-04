# Extending tenfabric — Step-by-Step Guides

Each guide includes exact file paths, code snippets, and tests to run.

---

## 1. Add a CLI Command

**Example**: Adding a `tfab logs` command to view training logs.

### Step 1: Create the command module
**File**: `src/tenfabric/cli/logs.py`
```python
from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

console = Console()


def logs_cmd(
    run_id: str = typer.Argument(..., help="Run ID to view logs for."),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output."),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show."),
) -> None:
    """View training logs for a run."""
    from tenfabric.core.run_store import RunStore

    store = RunStore()
    run = store.get(run_id)
    if not run:
        console.print(f"[red]Run not found:[/] {run_id}")
        raise typer.Exit(1)

    console.print(f"[bold]Logs for {run_id}[/]")
    # ... implementation
```

### Step 2: Register in app.py
**File**: `src/tenfabric/cli/app.py` — add near the other imports and registrations:
```python
from tenfabric.cli.logs import logs_cmd  # noqa: E402
app.command(name="logs", help="View training logs for a run.")(logs_cmd)
```

### Step 3: Add tests
**File**: `tests/test_cli.py` — add test class:
```python
class TestLogs:
    def test_logs_missing_run(self):
        result = runner.invoke(app, ["logs", "nonexistent-run"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()
```

**Files modified**: `cli/logs.py` (new), `cli/app.py`, `tests/test_cli.py`
**Tests to run**: `uv run pytest tests/test_cli.py -v`

---

## 2. Add a Training Backend

**Example**: Adding an Axolotl backend.

### Step 1: Add enum value
**File**: `src/tenfabric/config/schema.py`
```python
class TrainingBackend(str, Enum):
    TRL = "trl"
    UNSLOTH = "unsloth"
    AXOLOTL = "axolotl"  # ← add this
```

### Step 2: Create the backend module
**File**: `src/tenfabric/training/axolotl_backend.py`
```python
from __future__ import annotations

from typing import Any

from tenfabric.config.schema import TenfabricConfig


def prepare_model(config: TenfabricConfig) -> tuple[Any, Any]:
    """Load model using Axolotl."""
    try:
        from axolotl.utils.models import load_model_and_tokenizer
    except ImportError:
        raise RuntimeError(
            "Axolotl not installed. Install it with:\n"
            "  pip install axolotl\n"
            "Or switch backend: training.backend: trl"
        )
    # ... implementation
    return model, tokenizer


def train(config: TenfabricConfig, model: Any, tokenizer: Any, dataset: Any) -> None:
    """Run training using Axolotl."""
    # ... implementation
```

### Step 3: Add dispatch in pipeline.py
**File**: `src/tenfabric/core/pipeline.py` — update `_prepare_model` and `_train`:
```python
def _prepare_model(self, config: TenfabricConfig) -> None:
    backend = config.training.backend.value
    if backend == "trl":
        self._prepare_trl(config)
    elif backend == "unsloth":
        self._prepare_unsloth(config)
    elif backend == "axolotl":
        self._prepare_axolotl(config)

def _prepare_axolotl(self, config: TenfabricConfig) -> None:
    from tenfabric.training.axolotl_backend import prepare_model
    self._model, self._tokenizer = prepare_model(config)
```

Same pattern for `_train`.

### Step 4: Add dispatch in activities.py
**File**: `src/tenfabric/workflows/activities.py` — update `train_model` activity:
```python
if backend == "axolotl":
    from tenfabric.training.axolotl_backend import prepare_model, train
```

### Step 5: Add optional dependency
**File**: `pyproject.toml` — add new extra:
```toml
[project.optional-dependencies]
axolotl = ["axolotl>=0.4"]
```

**Files modified**: `config/schema.py`, `training/axolotl_backend.py` (new), `core/pipeline.py`, `workflows/activities.py`, `pyproject.toml`
**Tests to run**: `uv run pytest tests/ -v` (schema change → run all)

---

## 3. Add a Dataset Format

**Example**: Adding a `chatml` format.

### Step 1: Add enum value
**File**: `src/tenfabric/config/schema.py`
```python
class DatasetFormat(str, Enum):
    ALPACA = "alpaca"
    SHAREGPT = "sharegpt"
    CUSTOM = "custom"
    CHATML = "chatml"  # ← add this
```

### Step 2: Add formatter function
**File**: `src/tenfabric/training/data.py`
```python
def _format_chatml(example: dict) -> dict:
    """Format ChatML-style conversations into text field."""
    messages = example.get("messages", [])
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return {"text": "\n".join(parts)}
```

### Step 3: Add dispatch branch
**File**: `src/tenfabric/training/data.py` — in `load_and_format_dataset`:
```python
elif fmt == DatasetFormat.CHATML:
    ds = ds.map(_format_chatml, remove_columns=ds.column_names)
```

### Step 4: Add tests
**File**: `tests/test_data.py`
```python
class TestFormatChatML:
    def test_basic_conversation(self):
        example = {"messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]}
        result = _format_chatml(example)
        assert "text" in result
        assert "<|im_start|>user" in result["text"]
        assert "<|im_end|>" in result["text"]
```

**Files modified**: `config/schema.py`, `training/data.py`, `tests/test_data.py`
**Tests to run**: `uv run pytest tests/ -v` (schema change → run all)

---

## 4. Add an Infrastructure Provider

**Example**: Adding a Modal provider.

### Step 1: Add enum value
**File**: `src/tenfabric/config/schema.py`
```python
class InfraProvider(str, Enum):
    AUTO = "auto"
    LOCAL = "local"
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    RUNPOD = "runpod"
    LAMBDA = "lambda"
    MODAL = "modal"  # ← add this
```

### Step 2: Create provider module
**File**: `src/tenfabric/infra/modal.py`
```python
from __future__ import annotations

from tenfabric.config.schema import TenfabricConfig
from tenfabric.infra.base import InfraHandle


class ModalProvider:
    """Provision GPU instances via Modal."""

    def provision(self, config: TenfabricConfig) -> InfraHandle:
        try:
            import modal
        except ImportError:
            raise RuntimeError("Modal not installed: pip install modal")
        # ... implementation
        return InfraHandle(provider="modal", instance_id="...", status="ready")

    def setup(self, handle: InfraHandle, config: TenfabricConfig) -> None:
        pass  # Modal handles environment via image

    def teardown(self, handle: InfraHandle) -> None:
        pass  # Modal auto-tears down

    def status(self, handle: InfraHandle) -> str:
        return "ready"
```

### Step 3: Add dispatch in activities.py
**File**: `src/tenfabric/workflows/activities.py` — update `provision_infra`:
```python
elif config.infra.provider.value == "modal":
    from tenfabric.infra.modal import ModalProvider
    provider = ModalProvider()
    handle = provider.provision(config)
```

### Step 4: Add optional dependency
**File**: `pyproject.toml`:
```toml
[project.optional-dependencies]
modal = ["modal>=0.60"]
```

### Step 5: Add tests
**File**: `tests/test_infra.py`
```python
class TestModalProvider:
    def test_provision_returns_handle(self):
        # Mock modal import
        provider = ModalProvider()
        # ... test with mocks
```

**Files modified**: `config/schema.py`, `infra/modal.py` (new), `workflows/activities.py`, `pyproject.toml`, `tests/test_infra.py`
**Tests to run**: `uv run pytest tests/ -v` (schema change → run all)

---

## 5. Add a Training Method

**Example**: Adding SPIN (Self-Play Fine-Tuning).

### Step 1: Add enum value
**File**: `src/tenfabric/config/schema.py`
```python
class TrainingMethod(str, Enum):
    SFT = "sft"
    DPO = "dpo"
    GRPO = "grpo"
    PPO = "ppo"
    KTO = "kto"
    ORPO = "orpo"
    SPIN = "spin"  # ← add this
```

### Step 2: Add trainer in backend
**File**: `src/tenfabric/training/trl_backend.py` — add dispatch and implementation:
```python
elif method == TrainingMethod.SPIN:
    _train_spin(config, model, tokenizer, dataset)

def _train_spin(config, model, tokenizer, dataset):
    from trl import SPINTrainer, SPINConfig
    # ... configure and run trainer
```

**Files modified**: `config/schema.py`, `training/trl_backend.py` (and/or `unsloth_backend.py`)
**Tests to run**: `uv run pytest tests/ -v` (schema change → run all)
