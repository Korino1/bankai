"""Backend implementations for different model runtimes."""

from bankai.backends.base import Backend

__all__ = ["Backend", "get_backend"]


def get_backend(name: str) -> Backend:
    """Instantiate a backend by name."""
    if name == "mlx":
        from bankai.backends.mlx_backend import MLXBackend
        return MLXBackend()
    elif name == "gguf":
        from bankai.backends.gguf_backend import GGUFBackend
        return GGUFBackend()
    else:
        raise ValueError(f"Unknown backend: {name}. Available: mlx, gguf")
