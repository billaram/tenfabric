# Testing Guide — tenfabric

AI-specific testing reference. For human-readable test philosophy, see `TESTING.md` in the project root.

## Test Philosophy
- **No GPU required** — All 119 tests run on CPU in <1s
- **No network calls** — No HuggingFace downloads, no API calls
- **No Temporal server** — Core tests don't need a running workflow engine
- **Fast feedback** — Full suite completes in under 1 second
- **schema.py is sacred** — Any change to schema requires running ALL tests

## How to Run Tests

```bash
uv run pytest tests/ -v                          # Full suite (119 tests)
uv run pytest tests/test_config.py -v            # Single module
uv run pytest tests/test_cli.py::TestTrain -v    # Single class
uv run pytest tests/ -k "test_minimal" -v        # By name pattern
uv run pytest tests/ --tb=short                  # Shorter tracebacks
```

## Test File → Source Module Map

| Test File | Source Module(s) | Test Count |
|-----------|------------------|------------|
| `test_config.py` | config/schema.py, config/loader.py | 7 |
| `test_config_edge_cases.py` | config/schema.py, config/loader.py | 22 |
| `test_defaults.py` | config/defaults.py | 8 |
| `test_cli.py` | cli/* (all commands) | 26 |
| `test_data.py` | training/data.py | 9 |
| `test_infra.py` | infra/base.py, infra/local.py | 8 |
| `test_gpu_advisor.py` | infra/gpu_advisor.py | 7 |
| `test_skypilot_config.py` | infra/skypilot.py | 17 |
| `test_run_store.py` | core/run_store.py | 5 |

**Untested modules** (need GPU, heavy deps, or Temporal):
- `cli/status.py` — Easy to test (mock RunStore)
- `training/trl_backend.py` — Hard (needs torch mock)
- `training/unsloth_backend.py` — Hard (needs unsloth mock)
- `training/export.py` — Medium (mock model objects)
- `core/pipeline.py` — Medium (mock training backends)
- `workflows/*` — Hard (needs Temporal test environment)

## Change Impact Matrix

When you modify a file, these tests MUST pass:

| Changed File | Required Tests |
|-------------|----------------|
| `config/schema.py` | **ALL TESTS** (`uv run pytest tests/ -v`) |
| `config/loader.py` | `test_config.py`, `test_config_edge_cases.py` |
| `config/defaults.py` | `test_defaults.py`, `test_gpu_advisor.py` |
| `cli/app.py` | `test_cli.py` |
| `cli/train.py` | `test_cli.py::TestTrain` |
| `cli/init.py` | `test_cli.py::TestInit` |
| `cli/doctor.py` | `test_cli.py::TestDoctor` |
| `cli/cost.py` | `test_cli.py::TestCost` |
| `cli/models.py` | `test_cli.py::TestModels` |
| `cli/examples.py` | `test_cli.py::TestExamples` |
| `training/data.py` | `test_data.py` |
| `infra/base.py` | `test_infra.py` |
| `infra/local.py` | `test_infra.py` |
| `infra/skypilot.py` | `test_skypilot_config.py` |
| `infra/gpu_advisor.py` | `test_gpu_advisor.py` |
| `core/run_store.py` | `test_run_store.py` |

## Mocking Patterns

### Mock GPU detection
```python
from unittest.mock import patch, MagicMock

def test_with_gpu():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
    mock_torch.cuda.get_device_properties.return_value.total_mem = 24 * (1024**3)
    mock_torch.cuda.device_count.return_value = 1

    with patch.dict("sys.modules", {"torch": mock_torch}):
        # ... test GPU-dependent code

def test_without_gpu():
    with patch.dict("sys.modules", {"torch": None}):
        # ... test ImportError handling
```

### Mock training backends
```python
def test_pipeline_calls_backend(monkeypatch):
    mock_prepare = MagicMock(return_value=(MagicMock(), MagicMock()))
    mock_train = MagicMock()
    mock_export = MagicMock()

    monkeypatch.setattr("tenfabric.training.trl_backend.prepare_model", mock_prepare)
    monkeypatch.setattr("tenfabric.training.trl_backend.train", mock_train)
    monkeypatch.setattr("tenfabric.training.export.export_model", mock_export)
```

### Mock CLI commands
```python
from typer.testing import CliRunner
from tenfabric.cli.app import app

runner = CliRunner()

def test_train_dry_run(tmp_path):
    config_file = tmp_path / "tenfabric.yaml"
    config_file.write_text("project: test\nmodel:\n  base: test/model\n  ...")
    result = runner.invoke(app, ["train", str(config_file), "--dry-run"])
    assert result.exit_code == 0
    assert "Dry run" in result.output
```

### Mock config validation
```python
from tenfabric.config.schema import TenfabricConfig

def _make_config(**overrides):
    """Helper: create TenfabricConfig with defaults + overrides."""
    base = {
        "project": "test",
        "model": {"base": "test/model", "method": "lora", "quantization": "4bit"},
        "dataset": {"source": "test/data", "format": "alpaca"},
    }
    base.update(overrides)
    return TenfabricConfig(**base)

def test_validation():
    config = _make_config(training={"epochs": 5, "batch_size": 8})
    assert config.training.epochs == 5
```

### Mock SQLite with tmp_path
```python
from tenfabric.core.run_store import RunStore

def test_run_store(tmp_path):
    store = RunStore(db_path=tmp_path / "test.db")
    store.create(run_id="run-001", project="test", model="llama", provider="local")
    run = store.get("run-001")
    assert run["status"] == "pending"
```

## Coverage Gaps and Priority

| Module | Priority | Difficulty | Mock Strategy |
|--------|----------|------------|---------------|
| `cli/status.py` | High | Easy | Mock `RunStore` |
| `training/export.py` | High | Medium | Mock model objects, mock unsloth import |
| `core/pipeline.py` | High | Medium | Mock all training backends + export |
| `training/trl_backend.py` | Medium | Hard | Mock torch, transformers, peft, trl |
| `training/unsloth_backend.py` | Medium | Hard | Mock unsloth + trl |
| `workflows/activities.py` | Low | Hard | Temporal test harness |
| `workflows/client.py` | Low | Hard | Mock Temporal client + subprocess |
| `workflows/training_pipeline.py` | Low | Hard | Temporal workflow test environment |

## 5 Test Design Rules

1. **One assertion per concept** — A test should verify one behavior. Multiple asserts are fine if they test different aspects of the same output.
2. **Test the contract, not the implementation** — Test that `_format_alpaca` returns `{"text": ...}` with the right content, not that it calls `.get()` on the dict.
3. **Use `_make_config` helpers** — Don't copy-paste full config dicts. Override only what matters for the test.
4. **Use `tmp_path` for file operations** — Never write to the real filesystem. SQLite tests use `tmp_path / "test.db"`.
5. **Expect errors precisely** — Use `pytest.raises(ValidationError, match="specific message")`, not bare `pytest.raises(Exception)`.
