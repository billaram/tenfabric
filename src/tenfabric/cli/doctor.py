"""tfab doctor — environment diagnostics."""

from __future__ import annotations

import shutil
import subprocess
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def doctor_cmd() -> None:
    """Check your environment for GPU, CUDA, and dependencies."""
    checks: list[tuple[str, bool, str]] = []

    # Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 10)
    checks.append(("Python", py_ok, py_ver if py_ok else f"{py_ver} (need >=3.10)"))

    # CUDA
    cuda_ver = _check_cuda()
    checks.append(("CUDA", cuda_ver is not None, cuda_ver or "Not found"))

    # PyTorch
    torch_info = _check_torch()
    checks.append(("PyTorch", torch_info["installed"], torch_info["detail"]))

    # GPU
    gpu_info = _check_gpu()
    checks.append(("GPU", gpu_info["available"], gpu_info["detail"]))

    # Key packages
    for pkg, pip_name in [
        ("transformers", "transformers"),
        ("trl", "trl"),
        ("peft", "peft"),
        ("datasets", "datasets"),
        ("accelerate", "accelerate"),
        ("bitsandbytes", "bitsandbytes"),
    ]:
        installed, ver = _check_package(pkg)
        if installed:
            checks.append((pip_name, True, ver))
        else:
            checks.append((pip_name, False, f"Not installed → pip install {pip_name}"))

    # Optional: unsloth
    installed, ver = _check_package("unsloth")
    if installed:
        checks.append(("unsloth", True, ver))
    else:
        checks.append(("unsloth", False, "Not installed (optional) → pip install unsloth"))

    # Optional: skypilot
    installed, ver = _check_package("sky")
    if installed:
        checks.append(("skypilot", True, ver))
    else:
        checks.append(("skypilot", False, "Not installed (for cloud) → pip install 'skypilot[aws]'"))

    # Optional: temporal
    installed, ver = _check_package("temporalio")
    if installed:
        checks.append(("temporalio", True, ver))
    else:
        checks.append(("temporalio", False, "Not installed (for cloud) → pip install temporalio"))

    # Display results
    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("Component", style="bold")
    table.add_column("Status", width=4, justify="center")
    table.add_column("Detail")

    all_required_ok = True
    for name, ok, detail in checks:
        status = "[green]✓[/]" if ok else "[red]✗[/]"
        detail_style = "" if ok else "[yellow]"
        detail_end = "" if ok else "[/]"
        table.add_row(name, status, f"{detail_style}{detail}{detail_end}")
        if not ok and name in ("Python", "PyTorch", "GPU", "CUDA"):
            all_required_ok = False

    console.print(Panel(table, title="[bold]tfab doctor[/]", border_style="blue"))

    # Summary
    if all_required_ok:
        vram = gpu_info.get("vram_gb")
        if vram:
            if vram >= 24:
                console.print(f"\n[green]Ready for local training up to ~8B params (LoRA 4-bit)[/]")
            elif vram >= 16:
                console.print(f"\n[green]Ready for local training up to ~7B params (LoRA 4-bit)[/]")
            elif vram >= 8:
                console.print(f"\n[green]Ready for local training up to ~3B params (LoRA 4-bit)[/]")
            else:
                console.print(f"\n[yellow]Limited VRAM ({vram}GB). Consider cloud training for larger models.[/]")
        else:
            console.print("\n[green]Core dependencies look good.[/]")
    else:
        console.print("\n[red]Some required dependencies are missing. Install them to proceed.[/]")
        console.print("[dim]Tip: pip install 'tenfabric[training]'[/]")


def _check_cuda() -> str | None:
    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            result = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=5)
            for line in result.stdout.splitlines():
                if "release" in line.lower():
                    return line.strip().split("release")[-1].strip().rstrip(",")
        except Exception:
            pass

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run([nvidia_smi], capture_output=True, text=True, timeout=5)
            for line in result.stdout.splitlines():
                if "CUDA Version" in line:
                    parts = line.split("CUDA Version:")
                    if len(parts) > 1:
                        return parts[1].strip().split()[0]
        except Exception:
            pass

    return None


def _check_torch() -> dict:
    try:
        import torch
        ver = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_ver = torch.version.cuda
            return {"installed": True, "detail": f"{ver} (CUDA {cuda_ver})"}
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return {"installed": True, "detail": f"{ver} (MPS/Apple Silicon)"}
        else:
            return {"installed": True, "detail": f"{ver} (CPU only)"}
    except ImportError:
        return {"installed": False, "detail": "Not installed → pip install torch"}


def _check_gpu() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_mem
            vram_gb = round(vram_bytes / (1024**3), 1)
            count = torch.cuda.device_count()
            detail = f"{name} ({vram_gb}GB)"
            if count > 1:
                detail += f" x{count}"
            return {"available": True, "detail": detail, "vram_gb": vram_gb}
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return {"available": True, "detail": "Apple Silicon (MPS)", "vram_gb": None}
        else:
            return {"available": False, "detail": "No GPU detected"}
    except ImportError:
        return {"available": False, "detail": "PyTorch not installed — cannot detect GPU"}


def _check_package(name: str) -> tuple[bool, str]:
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "unknown")
        return True, str(ver)
    except ImportError:
        return False, ""
