"""Temporal client — start and manage training workflows."""

from __future__ import annotations

import asyncio
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
    address = temporal_address or config.workflow.temporal_address
    task_queue = config.workflow.task_queue or DEFAULT_TASK_QUEUE
    config_dict = config.model_dump(mode="json")

    if address:
        # Connect to an existing Temporal server
        return await _run_with_external_server(address, run_id, config_dict, task_queue)
    else:
        # Start an embedded dev server via WorkflowEnvironment.start_local()
        return await _run_with_dev_server(run_id, config_dict, task_queue)


async def _run_with_external_server(
    address: str, run_id: str, config_dict: dict, task_queue: str
) -> dict:
    """Run workflow against an existing Temporal server."""
    from temporalio.client import Client
    from temporalio.worker import Worker

    from tenfabric.workflows.activities import (
        export_model,
        prepare_dataset,
        provision_infra,
        setup_environment,
        teardown_infra,
        train_model,
        validate_config,
    )

    client = await Client.connect(address)

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[TrainingPipelineWorkflow],
        activities=[
            validate_config,
            provision_infra,
            setup_environment,
            prepare_dataset,
            train_model,
            export_model,
            teardown_infra,
        ],
    ):
        handle = await client.start_workflow(
            TrainingPipelineWorkflow.run,
            args=[run_id, config_dict],
            id=run_id,
            task_queue=task_queue,
        )
        console.print(f"  [dim]Temporal workflow started: {handle.id}[/]")
        return await handle.result()


async def _run_with_dev_server(
    run_id: str, config_dict: dict, task_queue: str
) -> dict:
    """Start an embedded Temporal dev server and run the workflow."""
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker

    from tenfabric.workflows.activities import (
        export_model,
        prepare_dataset,
        provision_infra,
        setup_environment,
        teardown_infra,
        train_model,
        validate_config,
    )

    console.print("  [dim]Starting Temporal dev server...[/]")

    try:
        env = await WorkflowEnvironment.start_local()
    except Exception as e:
        raise RuntimeError(
            f"Cannot start Temporal dev server: {e}\n"
            "Install temporalio with test server support: pip install temporalio\n"
            "Or point to an existing server: workflow.temporal_address in config"
        ) from e

    console.print("  [green]\u2713[/] Temporal dev server started")

    async with env:
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[TrainingPipelineWorkflow],
            activities=[
                validate_config,
                provision_infra,
                setup_environment,
                prepare_dataset,
                train_model,
                export_model,
                teardown_infra,
            ],
        ):
            handle = await env.client.start_workflow(
                TrainingPipelineWorkflow.run,
                args=[run_id, config_dict],
                id=run_id,
                task_queue=task_queue,
            )
            console.print(f"  [dim]Temporal workflow started: {handle.id}[/]")
            return await handle.result()
