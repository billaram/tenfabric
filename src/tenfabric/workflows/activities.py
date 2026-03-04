"""Temporal activity implementations for training pipeline steps."""

from __future__ import annotations

from temporalio import activity

from tenfabric.config.schema import InfraProvider, TenfabricConfig


@activity.defn
async def validate_config(config_dict: dict) -> dict:
    """Validate the training configuration."""
    config = TenfabricConfig(**config_dict)
    return {"valid": True, "project": config.project}


@activity.defn
async def provision_infra(config_dict: dict) -> dict:
    """Provision cloud infrastructure via SkyPilot."""
    config = TenfabricConfig(**config_dict)

    if config.infra.provider == InfraProvider.LOCAL:
        return {"provider": "local", "instance_id": "localhost", "status": "ready"}

    from tenfabric.infra.skypilot import SkyPilotProvider

    provider = SkyPilotProvider()
    handle = provider.provision(config)
    return {
        "provider": handle.provider,
        "instance_id": handle.instance_id,
        "host": handle.host,
        "status": handle.status,
        "metadata": handle.metadata,
    }


@activity.defn
async def setup_environment(infra_handle: dict, config_dict: dict) -> dict:
    """Install dependencies and prepare the training environment."""
    config = TenfabricConfig(**config_dict)

    if infra_handle.get("provider") == "local":
        return {"status": "ready"}

    # For cloud: SkyPilot handles this via setup script in the YAML
    return {"status": "ready"}


@activity.defn
async def prepare_dataset(infra_handle: dict, config_dict: dict) -> dict:
    """Load and preprocess the training dataset."""
    config = TenfabricConfig(**config_dict)
    activity.heartbeat("Loading dataset...")

    from tenfabric.training.data import load_and_format_dataset

    dataset = load_and_format_dataset(config)
    return {"num_samples": len(dataset), "status": "ready"}


@activity.defn
async def train_model(infra_handle: dict, config_dict: dict) -> dict:
    """Run the actual model training."""
    config = TenfabricConfig(**config_dict)

    if infra_handle.get("provider") == "local":
        # Run training locally
        activity.heartbeat("Preparing model...")

        backend = config.training.backend.value
        if backend == "trl":
            from tenfabric.training.trl_backend import prepare_model, train
        else:
            from tenfabric.training.unsloth_backend import prepare_model, train

        model, tokenizer = prepare_model(config)

        activity.heartbeat("Loading dataset...")
        from tenfabric.training.data import load_and_format_dataset

        dataset = load_and_format_dataset(config)

        activity.heartbeat("Training...")
        train(config, model, tokenizer, dataset)

        return {"status": "completed", "output_dir": config.output.dir}
    else:
        # For cloud: training runs on the remote instance via SkyPilot
        # Monitor by polling SkyPilot job status
        return {"status": "completed", "output_dir": config.output.dir}


@activity.defn
async def export_model(infra_handle: dict, config_dict: dict, train_result: dict) -> dict:
    """Export trained model — merge adapters, GGUF, Hub push."""
    config = TenfabricConfig(**config_dict)
    # Export is handled on the training instance
    return {"status": "exported", "output_dir": config.output.dir}


@activity.defn
async def teardown_infra(infra_handle: dict) -> dict:
    """Tear down provisioned infrastructure."""
    if infra_handle.get("provider") in ("local", None):
        return {"status": "skipped"}

    from tenfabric.infra.skypilot import SkyPilotProvider
    from tenfabric.infra.base import InfraHandle

    provider = SkyPilotProvider()
    handle = InfraHandle(
        provider=infra_handle["provider"],
        instance_id=infra_handle.get("instance_id"),
        metadata=infra_handle.get("metadata", {}),
    )
    provider.teardown(handle)
    return {"status": "torn_down"}
