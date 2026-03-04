"""Temporal client — start and manage training workflows."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from typing import Optional

from rich.console import Console

from tenfabric.config.schema import TenfabricConfig
from tenfabric.workflows.training_pipeline import TrainingPipelineWorkflow

console = Console()

DEFAULT_TASK_QUEUE = "tenfabric-training"


async def start_training_workflow(
    run_id: str,
    config: TenfabricConfig,
    temporal_address: str | None = None,
) -> dict:
    """Start a training workflow on Temporal and wait for completion."""
    from temporalio.client import Client

    address = temporal_address or config.workflow.temporal_address
    if not address:
        address = await _ensure_dev_server()

    # Start worker in background
    worker_proc = _start_worker_process(address, config.workflow.task_queue)

    try:
        client = await Client.connect(address)

        # Serialize config to dict for Temporal
        config_dict = config.model_dump(mode="json")

        # Start the workflow
        handle = await client.start_workflow(
            TrainingPipelineWorkflow.run,
            args=[run_id, config_dict],
            id=run_id,
            task_queue=config.workflow.task_queue or DEFAULT_TASK_QUEUE,
        )

        console.print(f"  [dim]Temporal workflow started: {handle.id}[/]")

        # Wait for completion
        result = await handle.result()
        return result

    finally:
        if worker_proc:
            worker_proc.terminate()
            worker_proc.wait(timeout=5)


async def _ensure_dev_server() -> str:
    """Start an embedded Temporal dev server if none is running."""
    address = "localhost:7233"

    # Check if Temporal is already running
    try:
        from temporalio.client import Client

        client = await Client.connect(address)
        await client.get_system_info()
        console.print("  [dim]Connected to existing Temporal server[/]")
        return address
    except Exception:
        pass

    # Start dev server
    console.print("  [dim]Starting Temporal dev server...[/]")

    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "temporalio.testing._workflow", "--port", "7233"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Give it a moment to start
        time.sleep(2)

        if proc.poll() is not None:
            raise RuntimeError("Dev server exited immediately")

        console.print("  [green]✓[/] Temporal dev server started")
        return address

    except FileNotFoundError:
        raise RuntimeError(
            "Cannot start Temporal dev server.\n"
            "Install temporalio: pip install temporalio\n"
            "Or point to an existing server: workflow.temporal_address in config"
        )


def _start_worker_process(
    temporal_address: str, task_queue: str
) -> subprocess.Popen | None:
    """Start a Temporal worker in a background process."""
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                f"from tenfabric.workflows.worker import start_worker; "
                f"start_worker('{temporal_address}', '{task_queue}')",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(1)
        if proc.poll() is not None:
            return None
        return proc
    except Exception:
        return None
