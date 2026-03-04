"""Temporal workflow definition for the training pipeline."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from tenfabric.config.schema import TenfabricConfig


@workflow.defn
class TrainingPipelineWorkflow:
    """Durable training pipeline — provision, setup, train, export, teardown.

    Each step is a Temporal activity with retry policies. If the cloud instance
    gets preempted (spot), Temporal automatically retries the activity.
    """

    @workflow.run
    async def run(self, run_id: str, config_dict: dict) -> dict:
        config = TenfabricConfig(**config_dict)

        # Step 1: Validate config
        await workflow.execute_activity(
            "validate_config",
            args=[config_dict],
            start_to_close_timeout=timedelta(seconds=30),
        )

        # Step 2: Provision infrastructure
        infra_handle = await workflow.execute_activity(
            "provision_infra",
            args=[config_dict],
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=workflow.RetryPolicy(
                maximum_attempts=config.workflow.retry_policy.max_attempts,
                initial_interval=timedelta(seconds=10),
                backoff_coefficient=2.0,
            ),
        )

        try:
            # Step 3: Setup environment
            await workflow.execute_activity(
                "setup_environment",
                args=[infra_handle, config_dict],
                start_to_close_timeout=timedelta(minutes=15),
                retry_policy=workflow.RetryPolicy(maximum_attempts=2),
            )

            # Step 4: Load and prepare dataset
            await workflow.execute_activity(
                "prepare_dataset",
                args=[infra_handle, config_dict],
                start_to_close_timeout=timedelta(minutes=30),
                retry_policy=workflow.RetryPolicy(maximum_attempts=2),
            )

            # Step 5: Train model
            train_result = await workflow.execute_activity(
                "train_model",
                args=[infra_handle, config_dict],
                start_to_close_timeout=timedelta(hours=24),
                heartbeat_timeout=timedelta(minutes=5),
                retry_policy=workflow.RetryPolicy(
                    maximum_attempts=config.workflow.retry_policy.max_attempts,
                    initial_interval=timedelta(seconds=30),
                    backoff_coefficient=2.0,
                ),
            )

            # Step 6: Export model
            export_result = await workflow.execute_activity(
                "export_model",
                args=[infra_handle, config_dict, train_result],
                start_to_close_timeout=timedelta(minutes=30),
                retry_policy=workflow.RetryPolicy(maximum_attempts=2),
            )

            return {
                "run_id": run_id,
                "status": "completed",
                "train_result": train_result,
                "export_result": export_result,
            }

        finally:
            # Step 7: Always teardown infrastructure
            await workflow.execute_activity(
                "teardown_infra",
                args=[infra_handle],
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=workflow.RetryPolicy(maximum_attempts=3),
            )
