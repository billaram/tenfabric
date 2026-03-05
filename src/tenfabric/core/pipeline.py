"""Training pipeline orchestration — local and Temporal-backed."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from tenfabric.config.schema import TenfabricConfig
from tenfabric.core.run_store import RunStore

console = Console()


class LocalPipeline:
    """Run training steps sequentially in-process. No Temporal required."""

    def __init__(self) -> None:
        self.store = RunStore()

    def run(self, config: TenfabricConfig) -> None:
        run_id = _generate_run_id()
        self.store.create(
            run_id=run_id,
            project=config.project,
            model=config.model.base,
            provider=config.infra.provider.value,
            status="training",
        )

        steps = [
            ("Validating config", self._validate),
            ("Detecting GPU", self._detect_gpu),
            ("Loading dataset", self._load_dataset),
            ("Preparing model", self._prepare_model),
            ("Training", self._train),
            ("Exporting model", self._export),
        ]

        console.print(f"\n[bold cyan]Run:[/] {run_id}\n")

        try:
            for i, (name, step_fn) in enumerate(steps, 1):
                with Live(
                    Spinner("dots", text=Text(f"  Step {i}/{len(steps)}: {name}...")),
                    console=console,
                    transient=True,
                ):
                    step_fn(config)
                console.print(f"  [green]✓[/] Step {i}/{len(steps)}: {name}")

            self.store.update(run_id, status="completed")
            console.print(f"\n[bold green]Training complete![/]")
            console.print(f"  Output: {config.output.dir}")
            console.print(f"  Run ID: {run_id}\n")

        except Exception as e:
            self.store.update(run_id, status="failed", error=str(e))
            console.print(f"\n[bold red]Training failed:[/] {e}")
            console.print(f"  Run ID: {run_id}")
            console.print(f"  Check logs: tfab logs {run_id}\n")
            raise

    def _validate(self, config: TenfabricConfig) -> None:
        # Config already validated by Pydantic, but we can add runtime checks
        pass

    def _detect_gpu(self, config: TenfabricConfig) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                console.print(f"    [dim]GPU: {name} ({vram:.0f}GB)[/]")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                console.print("    [dim]GPU: Apple Silicon (MPS)[/]")
            else:
                console.print("    [yellow]Warning: No GPU detected. Training will be slow.[/]")
        except ImportError:
            raise RuntimeError(
                "PyTorch not installed. Install training dependencies:\n"
                "  pip install 'tenfabric[training]'"
            )

    def _load_dataset(self, config: TenfabricConfig) -> None:
        try:
            from datasets import load_dataset

            ds = load_dataset(
                config.dataset.source,
                split=config.dataset.split,
                trust_remote_code=True,
            )
            if config.dataset.max_samples and len(ds) > config.dataset.max_samples:
                ds = ds.select(range(config.dataset.max_samples))
            console.print(f"    [dim]Dataset: {len(ds)} samples[/]")
            # Store in instance for use in training step
            self._dataset = ds
        except ImportError:
            raise RuntimeError(
                "datasets library not installed:\n"
                "  pip install 'tenfabric[training]'"
            )

    def _prepare_model(self, config: TenfabricConfig) -> None:
        backend = config.training.backend.value
        if backend == "trl":
            self._prepare_trl(config)
        else:
            self._prepare_unsloth(config)

    def _prepare_trl(self, config: TenfabricConfig) -> None:
        from tenfabric.training.trl_backend import prepare_model

        self._model, self._tokenizer = prepare_model(config)

    def _prepare_unsloth(self, config: TenfabricConfig) -> None:
        from tenfabric.training.unsloth_backend import prepare_model

        self._model, self._tokenizer = prepare_model(config)

    def _train(self, config: TenfabricConfig) -> None:
        backend = config.training.backend.value
        if backend == "trl":
            from tenfabric.training.trl_backend import train

            train(config, self._model, self._tokenizer, self._dataset)
        else:
            from tenfabric.training.unsloth_backend import train

            train(config, self._model, self._tokenizer, self._dataset)

    def _export(self, config: TenfabricConfig) -> None:
        from tenfabric.training.export import export_model

        export_model(config, self._model, self._tokenizer)


class CloudPipeline:
    """Run training on a cloud GPU via SkyPilot. No Temporal required."""

    def __init__(self) -> None:
        self.store = RunStore()

    def run(self, config: TenfabricConfig) -> None:
        from tenfabric.infra.skypilot import SkyPilotProvider

        run_id = _generate_run_id()
        self.store.create(
            run_id=run_id,
            project=config.project,
            model=config.model.base,
            provider=config.infra.provider.value,
            status="provisioning",
        )

        provider = SkyPilotProvider()

        console.print(f"\n[bold cyan]Run:[/] {run_id}\n")

        try:
            # Provision + run (SkyPilot runs setup & training via cloud re-entry)
            # No spinner — stdout streams directly so user sees live logs
            console.print("  [bold]Step 1/2: Provisioning cloud GPU + training...[/]\n")
            handle = provider.provision(config)
            console.print("\n  [green]\u2713[/] Step 1/2: Provisioning cloud GPU")
            console.print(f"    [dim]Cluster: {handle.instance_id}[/]")

            self.store.update(run_id, status="completed")
            console.print(f"\n[bold green]Cloud training complete![/]")
            console.print(f"  Output: {config.output.dir}")
            console.print(f"  Run ID: {run_id}")
            console.print(f"  Cluster: {handle.instance_id}")
            console.print(f"  [dim]Teardown: sky down {handle.instance_id}[/]\n")

        except Exception as e:
            self.store.update(run_id, status="failed", error=str(e))
            console.print(f"\n[bold red]Cloud training failed:[/] {e}")
            console.print(f"  Run ID: {run_id}\n")
            raise

    def _provision(self, provider: object, config: TenfabricConfig) -> None:
        pass  # called inline above

    def _teardown(self, provider: object, config: TenfabricConfig) -> None:
        pass  # autostop handles this


class TemporalPipeline:
    """Run training via Temporal workflow for durable execution with retries."""

    def __init__(self) -> None:
        self.store = RunStore()

    def run(self, config: TenfabricConfig) -> None:
        import asyncio

        asyncio.run(self._run_async(config))

    async def _run_async(self, config: TenfabricConfig) -> None:
        from tenfabric.workflows.client import start_training_workflow

        run_id = _generate_run_id()
        self.store.create(
            run_id=run_id,
            project=config.project,
            model=config.model.base,
            provider=config.infra.provider.value,
            status="pending",
        )

        console.print(f"\n[bold cyan]Run:[/] {run_id}")
        console.print(f"[dim]Starting Temporal workflow...[/]\n")

        try:
            await start_training_workflow(run_id, config)
            console.print(f"\n[bold green]Workflow completed![/]")
            console.print(f"  Run ID: {run_id}")
            console.print(f"  Status: tfab status {run_id}\n")
        except Exception as e:
            self.store.update(run_id, status="failed", error=str(e))
            console.print(f"\n[bold red]Workflow failed:[/] {e}")
            raise


def _generate_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"run-{timestamp}-{short_uuid}"
