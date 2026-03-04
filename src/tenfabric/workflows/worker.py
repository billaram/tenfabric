"""Temporal worker — runs activities for the training pipeline."""

from __future__ import annotations

import asyncio

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
from tenfabric.workflows.training_pipeline import TrainingPipelineWorkflow

DEFAULT_TASK_QUEUE = "tenfabric-training"


async def run_worker(
    temporal_address: str = "localhost:7233",
    task_queue: str = DEFAULT_TASK_QUEUE,
) -> None:
    """Start a Temporal worker that processes training pipeline workflows."""
    client = await Client.connect(temporal_address)

    worker = Worker(
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
    )

    await worker.run()


def start_worker(
    temporal_address: str = "localhost:7233",
    task_queue: str = DEFAULT_TASK_QUEUE,
) -> None:
    """Entry point to start the worker (blocking)."""
    asyncio.run(run_worker(temporal_address, task_queue))
