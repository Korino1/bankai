"""
MLX backend for Bankai — runs on Apple Silicon using PrismML's MLX fork
with 1-bit kernel support.
"""

import numpy as np
import mlx.core as mx

from bankai.backends.base import Backend


def _get_module(model, path: str):
    """Navigate dotted path like 'model.layers.0.mlp.gate_proj'."""
    obj = model
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


class MLXBackend(Backend):
    """MLX + PrismML 1-bit kernels. Apple Silicon only."""

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load(self, model_path: str) -> None:
        from mlx_lm import load as mlx_load
        self.model, self.tokenizer = mlx_load(model_path)
        mx.eval(self.model.parameters())

    def num_layers(self) -> int:
        return len(self.model.model.layers)

    def _projection(self, layer: int, proj: str):
        path = f"model.layers.{layer}.mlp.{proj}"
        return _get_module(self.model, path)

    def num_rows(self, layer: int, proj: str) -> int:
        return int(self._projection(layer, proj).weight.shape[0])

    def get_row_scales(self, layer: int, proj: str) -> np.ndarray:
        mod = self._projection(layer, proj)
        return np.array(mx.mean(mx.abs(mod.scales), axis=1))

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text))

    def encode_token(self, text: str) -> int:
        return int(self.tokenizer.encode(text)[-1])

    def logit_gap(self, prompt_tokens: list[int], correct_id: int, wrong_id: int) -> float:
        tokens = mx.array(prompt_tokens)
        logits = self.model(tokens[None, :])
        last = logits[0, -1, :]
        mx.eval(last)
        return float(last[correct_id].item() - last[wrong_id].item())

    def flip_row(self, layer: int, proj: str, row: int) -> None:
        path = f"model.layers.{layer}.mlp.{proj}"
        mod = _get_module(self.model, path)
        w = mod.weight
        mask = mx.zeros_like(w)
        ones = mx.full((w.shape[1],), 0xFFFFFFFF, dtype=mx.uint32)
        mask = mask.at[row].add(ones)
        new_w = w ^ mask
        self.model.load_weights([(f"{path}.weight", new_w)], strict=False)
        mx.eval(self.model.parameters())

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        from mlx_lm import generate as mlx_generate
        return mlx_generate(self.model, self.tokenizer, prompt=prompt,
                            max_tokens=max_tokens, verbose=False)
