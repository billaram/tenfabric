"""Tests for infrastructure providers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tenfabric.config.schema import TenfabricConfig
from tenfabric.infra.base import InfraHandle
from tenfabric.infra.local import LocalProvider


def _make_config(**overrides) -> TenfabricConfig:
    base = {
        "project": "test",
        "model": {"base": "unsloth/Llama-3.2-1B", "method": "lora", "quantization": "4bit"},
        "dataset": {"source": "test-ds"},
        "infra": {"provider": "local"},
    }
    base.update(overrides)
    return TenfabricConfig(**base)


class TestLocalProvider:
    @patch("tenfabric.infra.local._detect_local_gpu")
    def test_provision_with_gpu(self, mock_detect):
        mock_detect.return_value = ("RTX 4090", 1)
        provider = LocalProvider()
        config = _make_config()
        handle = provider.provision(config)
        assert handle.provider == "local"
        assert handle.status == "ready"
        assert handle.gpu_name == "RTX 4090"
        assert handle.gpu_count == 1

    @patch("tenfabric.infra.local._detect_local_gpu")
    def test_provision_no_gpu(self, mock_detect):
        mock_detect.return_value = (None, 0)
        provider = LocalProvider()
        config = _make_config()
        handle = provider.provision(config)
        assert handle.provider == "local"
        assert handle.gpu_name is None
        assert handle.gpu_count == 0

    @patch("tenfabric.infra.local._detect_local_gpu")
    def test_provision_mps(self, mock_detect):
        mock_detect.return_value = ("Apple Silicon (MPS)", 1)
        provider = LocalProvider()
        config = _make_config()
        handle = provider.provision(config)
        assert handle.gpu_name == "Apple Silicon (MPS)"

    def test_setup_noop(self):
        provider = LocalProvider()
        handle = InfraHandle(provider="local", status="ready")
        provider.setup(handle, _make_config())  # Should not raise

    def test_teardown_noop(self):
        provider = LocalProvider()
        handle = InfraHandle(provider="local", status="ready")
        provider.teardown(handle)  # Should not raise

    def test_status(self):
        provider = LocalProvider()
        handle = InfraHandle(provider="local", status="ready")
        assert provider.status(handle) == "ready"


class TestInfraHandle:
    def test_defaults(self):
        handle = InfraHandle(provider="test")
        assert handle.instance_id is None
        assert handle.host is None
        assert handle.gpu_count == 1
        assert handle.status == "pending"
        assert handle.metadata == {}

    def test_full_handle(self):
        handle = InfraHandle(
            provider="skypilot",
            instance_id="cluster-123",
            host="1.2.3.4",
            port=22,
            gpu_name="A100",
            gpu_count=8,
            status="ready",
            metadata={"region": "us-east-1"},
        )
        assert handle.gpu_count == 8
        assert handle.metadata["region"] == "us-east-1"
