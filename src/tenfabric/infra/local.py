"""Local infrastructure provider — uses the current machine's GPU."""

from __future__ import annotations

from tenfabric.config.schema import TenfabricConfig
from tenfabric.infra.base import InfraHandle


class LocalProvider:
    """Uses the local machine's GPU for training. No provisioning needed."""

    def provision(self, config: TenfabricConfig) -> InfraHandle:
        gpu_name, gpu_count = _detect_local_gpu()
        return InfraHandle(
            provider="local",
            instance_id="localhost",
            host="localhost",
            gpu_name=gpu_name,
            gpu_count=gpu_count,
            status="ready",
        )

    def setup(self, handle: InfraHandle, config: TenfabricConfig) -> None:
        # Local setup is a no-op — user manages their own environment
        pass

    def teardown(self, handle: InfraHandle) -> None:
        # Nothing to tear down locally
        pass

    def status(self, handle: InfraHandle) -> str:
        return "ready"


def _detect_local_gpu() -> tuple[str | None, int]:
    """Detect local GPU name and count."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            count = torch.cuda.device_count()
            return name, count
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple Silicon (MPS)", 1
    except ImportError:
        pass
    return None, 0
