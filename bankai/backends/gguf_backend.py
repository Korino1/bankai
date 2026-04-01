"""
GGUF backend for Bankai — runs on any platform with PrismML's llama.cpp fork
compiled with CUDA (or Metal, or CPU) support.

Requires llama-cpp-python linked against the PrismML fork that provides Q1_0
kernels. Weight manipulation is done via direct memory access to the loaded
GGUF tensors.
"""

import ctypes
import os
from pathlib import Path

import numpy as np

from bankai.backends.base import Backend


class GGUFBackend(Backend):
    """llama.cpp + CUDA 1-bit kernels. NVIDIA GPUs, Apple Metal, or CPU."""

    def __init__(self):
        self.llm = None
        self._model_path = None
        self._row_flipped: dict[tuple[int, str, int], bool] = {}

    def load(self, model_path: str) -> None:
        """Load a Bonsai 8B GGUF model.

        model_path can be:
          - A local path to a .gguf file
          - A HuggingFace repo like "prism-ml/Bonsai-8B-gguf" (auto-downloads)
        """
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "GGUF backend requires llama-cpp-python built against PrismML's "
                "llama.cpp fork with Q1_0 kernels. See paper/bankai.pdf for "
                "setup instructions."
            ) from e

        # Resolve HF repo to local path if needed
        if not os.path.exists(model_path):
            from huggingface_hub import hf_hub_download
            # Assume default GGUF filename; caller can pass full path if different
            model_path = hf_hub_download(
                repo_id=model_path,
                filename="Bonsai-8B-Q1_0.gguf",
            )

        self._model_path = model_path
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=-1,  # offload all layers to GPU
            logits_all=False,
            verbose=False,
        )

    def num_layers(self) -> int:
        # Bonsai 8B has 36 transformer layers
        return 36

    def num_rows(self, layer: int, proj: str) -> int:
        # MLP dimensions for Bonsai 8B (Qwen3 architecture):
        # gate_proj, up_proj: (intermediate=12288, hidden=4096) → 12288 output rows
        # down_proj:          (hidden=4096, intermediate=12288) → 4096 output rows
        if proj in ("gate_proj", "up_proj"):
            return 12288
        elif proj == "down_proj":
            return 4096
        else:
            raise ValueError(f"Unknown projection: {proj}")

    def get_row_scales(self, layer: int, proj: str) -> np.ndarray:
        """Return per-row average scale magnitude from the GGUF Q1_0 tensors.

        TODO: This requires access to the quantization scale blocks inside
        llama.cpp's loaded model. Implementation deferred until we have the
        Python bindings exposing tensor memory.
        """
        raise NotImplementedError(
            "GGUF backend: get_row_scales not yet implemented. "
            "Requires direct access to Q1_0 scale blocks via llama.cpp bindings."
        )

    def encode(self, text: str) -> list[int]:
        return list(self.llm.tokenize(text.encode("utf-8"), add_bos=False))

    def encode_token(self, text: str) -> int:
        return int(self.llm.tokenize(text.encode("utf-8"), add_bos=False)[-1])

    def logit_gap(self, prompt_tokens: list[int], correct_id: int, wrong_id: int) -> float:
        # Reset context and evaluate the prompt
        self.llm.reset()
        self.llm.eval(prompt_tokens)
        # Get logits at the last position
        logits = self.llm._scores[-1] if hasattr(self.llm, "_scores") else self.llm.eval_logits[-1]
        return float(logits[correct_id] - logits[wrong_id])

    def flip_row(self, layer: int, proj: str, row: int) -> None:
        """XOR all bits in a row of a Q1_0 tensor in the loaded model.

        TODO: Implement via ctypes pointer to the weight tensor's GPU memory.
        This is the core unblocker for running Bankai on CUDA.
        """
        raise NotImplementedError(
            "GGUF backend: flip_row not yet implemented. "
            "Requires ctypes access to the Q1_0 weight tensor memory in llama.cpp."
        )

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        out = self.llm(prompt, max_tokens=max_tokens, echo=False)
        return out["choices"][0]["text"]
