"""
Modal runner: ship a Bankai search job to Modal's serverless GPU infrastructure.

Requires:
    pip install modal
    modal setup  # one-time authentication

Usage:
    from bankai.runners.modal_runner import run_modal_search
    patch = run_modal_search(
        backend="gguf",
        model_path="prism-ml/Bonsai-8B-gguf",
        target_probes=[...],
        control_probes=[...],
        max_iters=300,
    )
"""

from dataclasses import asdict
from typing import Optional

try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

from bankai.patch import Patch, PatchFlip
from bankai.probes import Probe


# ── Modal app definition ──

if MODAL_AVAILABLE:
    app = modal.App("bankai-search")

    # Image with PrismML's llama.cpp fork and CUDA support
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "cmake", "build-essential", "ninja-build", "wget")
        .run_commands(
            # Clone PrismML's llama.cpp fork with Q1_0 kernels
            "git clone --depth 1 https://github.com/PrismML-Eng/llama.cpp /root/llama.cpp",
            # Build with CUDA support
            "cd /root/llama.cpp && cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=OFF && cmake --build build -j --config Release",
        )
        .pip_install(
            "numpy",
            "huggingface_hub",
            "llama-cpp-python",  # will rebuild against our fork via env vars if needed
        )
        # Mount the bankai package into the container
        .add_local_python_source("bankai")
    )

    @app.function(
        image=image,
        gpu="H100",
        timeout=3600,  # 1 hour max per search
        memory=16384,
    )
    def _run_search_remote(config: dict) -> dict:
        """Execute a Bankai search entirely on Modal. Returns the patch as a dict."""
        from bankai.backends import get_backend
        from bankai.search import greedy_search
        from bankai.probes import Probe

        # Rehydrate probes from dicts
        target_probes = [Probe(**p) for p in config["target_probes"]]
        control_probes = [Probe(**p) for p in config["control_probes"]]

        backend = get_backend(config["backend"])
        backend.load(config["model_path"])

        patch = greedy_search(
            backend,
            target_probes=target_probes,
            control_probes=control_probes,
            search_layers=config.get("search_layers"),
            search_projs=config.get("search_projs"),
            max_iters=config.get("max_iters", 300),
            control_penalty=config.get("control_penalty", 2.0),
            fitness_mode=config.get("fitness_mode", "mean"),
            seed=config.get("seed", 42),
            patch_name=config.get("patch_name", "modal_search"),
            patch_description=config.get("patch_description", ""),
            base_model=config.get("base_model", "prism-ml/Bonsai-8B"),
        )

        # Serialize patch to dict (flips are PatchFlip objects)
        return {
            "name": patch.name,
            "description": patch.description,
            "base_model": patch.base_model,
            "flips": [asdict(f) for f in patch.flips],
            "metadata": patch.metadata,
        }


def run_modal_search(
    backend: str,
    model_path: str,
    target_probes: list[Probe],
    control_probes: list[Probe],
    search_layers: Optional[list[int]] = None,
    search_projs: Optional[list[str]] = None,
    max_iters: int = 300,
    control_penalty: float = 2.0,
    fitness_mode: str = "mean",
    seed: int = 42,
    patch_name: str = "modal_search",
    patch_description: str = "",
) -> Patch:
    """Run a Bankai search on Modal serverless infrastructure.

    This ships the entire search job to Modal (not per-iteration calls),
    runs it on an H100, and returns the final patch. Progress is streamed
    to the local terminal via Modal's built-in logging.
    """
    if not MODAL_AVAILABLE:
        raise ImportError(
            "Modal runner requires 'modal' package. Install with: pip install modal"
        )

    config = {
        "backend": backend,
        "model_path": model_path,
        "target_probes": [asdict(p) for p in target_probes],
        "control_probes": [asdict(p) for p in control_probes],
        "search_layers": search_layers,
        "search_projs": search_projs,
        "max_iters": max_iters,
        "control_penalty": control_penalty,
        "fitness_mode": fitness_mode,
        "seed": seed,
        "patch_name": patch_name,
        "patch_description": patch_description,
    }

    print(f"[modal] Launching search on H100 (backend={backend}, iters={max_iters})")
    with app.run():
        result = _run_search_remote.remote(config)
    print(f"[modal] Search complete — {len(result['flips'])} flips")

    return Patch(
        name=result["name"],
        description=result["description"],
        base_model=result["base_model"],
        flips=[PatchFlip(**f) for f in result["flips"]],
        metadata=result.get("metadata", {}),
    )
