"""SkyPilot infrastructure provider — multi-cloud GPU provisioning."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import yaml
from rich.console import Console

from tenfabric.config.defaults import estimate_vram, guess_model_size, recommend_gpu
from tenfabric.config.schema import InfraProvider as InfraProviderEnum
from tenfabric.config.schema import TenfabricConfig
from tenfabric.infra.base import InfraHandle

console = Console()

# Map tenfabric provider names to SkyPilot cloud names
PROVIDER_TO_SKY_CLOUD: dict[str, str] = {
    "aws": "aws",
    "gcp": "gcp",
    "azure": "azure",
    "runpod": "runpod",
    "lambda": "lambda",
}


class SkyPilotProvider:
    """Provision GPU instances via SkyPilot across multiple clouds."""

    def provision(self, config: TenfabricConfig) -> InfraHandle:
        _check_skypilot_installed()

        sky_yaml = _generate_sky_yaml(config)
        sky_yaml_path = _write_sky_yaml(sky_yaml, config.project)

        console.print(f"    [dim]SkyPilot config: {sky_yaml_path}[/]")

        # Launch via SkyPilot CLI
        result = subprocess.run(
            ["sky", "launch", str(sky_yaml_path), "-y", "--cluster", config.project],
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            raise RuntimeError(f"SkyPilot launch failed:\n{result.stderr}")

        return InfraHandle(
            provider="skypilot",
            instance_id=config.project,
            status="ready",
            metadata={"sky_yaml_path": str(sky_yaml_path)},
        )

    def setup(self, handle: InfraHandle, config: TenfabricConfig) -> None:
        # SkyPilot handles setup via the 'setup' section in the YAML
        pass

    def teardown(self, handle: InfraHandle) -> None:
        cluster_name = handle.instance_id
        if not cluster_name:
            return

        try:
            subprocess.run(
                ["sky", "down", cluster_name, "-y"],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except Exception as e:
            console.print(f"    [yellow]Warning: Failed to tear down cluster: {e}[/]")
            console.print(f"    [dim]Manual cleanup: sky down {cluster_name}[/]")

    def status(self, handle: InfraHandle) -> str:
        cluster_name = handle.instance_id
        try:
            result = subprocess.run(
                ["sky", "status", cluster_name, "--format", "json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if data:
                    return data[0].get("status", "unknown")
        except Exception:
            pass
        return "unknown"


def _check_skypilot_installed() -> None:
    try:
        import sky  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "SkyPilot not installed. Install cloud dependencies:\n"
            "  pip install 'tenfabric[cloud]'\n\n"
            "Then configure your cloud credentials:\n"
            "  sky check"
        )


def _generate_sky_yaml(config: TenfabricConfig) -> dict:
    """Generate a SkyPilot task YAML from tenfabric config."""
    sky_config: dict = {
        "name": config.project,
    }

    # Resources
    resources: dict = {}

    # GPU selection
    gpu = config.infra.gpu
    if gpu == "auto":
        gpu = _auto_select_gpu(config)
    if gpu:
        resources["accelerators"] = f"{gpu}:{config.infra.gpu_count}"

    # Spot instances
    resources["use_spot"] = config.infra.spot

    # Cloud/region
    provider = config.infra.provider
    if provider != InfraProviderEnum.AUTO:
        cloud = PROVIDER_TO_SKY_CLOUD.get(provider.value)
        if cloud:
            resources["cloud"] = cloud
    if config.infra.region != "auto":
        resources["region"] = config.infra.region

    # Disk
    resources["disk_size"] = config.infra.disk_size

    sky_config["resources"] = resources

    # Setup script — install tenfabric and training deps
    setup_lines = [
        "pip install 'tenfabric[training]'",
    ]
    if config.training.backend.value == "unsloth":
        setup_lines.append("pip install 'tenfabric[unsloth]'")

    if config.infra.skypilot.setup:
        setup_lines.append(config.infra.skypilot.setup)

    sky_config["setup"] = " && ".join(setup_lines)

    # Run command — execute tenfabric training
    sky_config["run"] = "tfab train /tmp/tenfabric-config.yaml --local"

    # File mounts — upload the config
    file_mounts = {"/tmp/tenfabric-config.yaml": "./tenfabric.yaml"}
    file_mounts.update(config.infra.skypilot.file_mounts)
    sky_config["file_mounts"] = file_mounts

    # Environment variables
    if config.infra.skypilot.envs:
        sky_config["envs"] = config.infra.skypilot.envs

    return sky_config


def _write_sky_yaml(sky_config: dict, project: str) -> Path:
    """Write SkyPilot YAML to a temp file."""
    sky_dir = Path.home() / ".tenfabric" / "skypilot"
    sky_dir.mkdir(parents=True, exist_ok=True)
    sky_path = sky_dir / f"{project}.yaml"
    with open(sky_path, "w") as f:
        yaml.dump(sky_config, f, default_flow_style=False)
    return sky_path


def _auto_select_gpu(config: TenfabricConfig) -> str | None:
    """Auto-select the best GPU based on model requirements."""
    model_size = guess_model_size(config.model.base)
    if not model_size:
        return "A10G"  # safe default

    vram_needed = estimate_vram(
        model_size, config.model.method.value, config.model.quantization.value
    )
    suitable = recommend_gpu(vram_needed)

    # Prefer cost-effective cloud GPUs
    cloud_preference = ["L4", "A10G", "T4", "A100-40GB", "A100-80GB", "H100"]
    for gpu in cloud_preference:
        if gpu in suitable:
            return gpu

    return suitable[0] if suitable else "A10G"
