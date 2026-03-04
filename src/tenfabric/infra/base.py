"""Infrastructure provider protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from tenfabric.config.schema import TenfabricConfig


@dataclass
class InfraHandle:
    """Handle to provisioned infrastructure."""

    provider: str
    instance_id: str | None = None
    host: str | None = None
    port: int | None = None
    gpu_name: str | None = None
    gpu_count: int = 1
    status: str = "pending"
    metadata: dict = field(default_factory=dict)


class InfraProvider(Protocol):
    """Protocol for infrastructure providers."""

    def provision(self, config: TenfabricConfig) -> InfraHandle:
        """Provision compute resources and return a handle."""
        ...

    def setup(self, handle: InfraHandle, config: TenfabricConfig) -> None:
        """Install dependencies and prepare the environment on provisioned infra."""
        ...

    def teardown(self, handle: InfraHandle) -> None:
        """Release provisioned resources."""
        ...

    def status(self, handle: InfraHandle) -> str:
        """Check the status of provisioned infrastructure."""
        ...
