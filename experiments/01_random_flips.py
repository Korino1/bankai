"""
Bankai Experiment 1: XOR Patch Viability
========================================
Tests the core thesis: can targeted bit flips in Bonsai 8B's binary weights
produce measurable behavioral change, and is the effect structured (not random)?

We measure perplexity on a fixed eval set after flipping N bits in:
  1. Random flips across ALL MLP layers
  2. Random flips in MLP layers 16-24 only (middle layers)
  3. Scale-guided flips (medium-scale groups in layers 16-24)
  4. Scale-guided flips (HIGH-scale groups — should break things faster)

If scale-guided flips produce MORE perplexity change per flip than random,
the thesis holds: targeted XOR patches can efficiently modify behavior.
"""

import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load

MODEL_PATH = "models/bonsai-8b-mlx"

EVAL_TEXTS = [
    "The capital of France is Paris, which is known for the Eiffel Tower.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "In quantum mechanics, the wave function describes the quantum state of a particle.",
    "To make pasta, boil water, add salt, cook for 8 minutes, then drain.",
    "The mitochondria is the powerhouse of the cell, producing ATP through oxidative phosphorylation.",
]


def compute_perplexity(model, tokenizer, texts: list[str]) -> float:
    """Compute average perplexity over a list of texts."""
    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        tokens = mx.array(tokenizer.encode(text))
        if tokens.shape[0] < 2:
            continue

        inputs = tokens[:-1][None, :]
        targets = tokens[1:]

        logits = model(inputs)
        logits = logits.squeeze(0)

        loss = nn.losses.cross_entropy(logits, targets, reduction="sum")
        mx.eval(loss)
        total_loss += loss.item()
        total_tokens += targets.shape[0]

    return float(np.exp(total_loss / total_tokens))


def get_mlp_weight_paths(model, layer_range: tuple[int, int] | None = None) -> list[str]:
    """Get dotted paths to MLP QuantizedLinear weight tensors."""
    paths = []
    for i, layer in enumerate(model.model.layers):
        if layer_range and not (layer_range[0] <= i <= layer_range[1]):
            continue
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            paths.append(f"model.layers.{i}.mlp.{proj}")
    return paths


def get_weight_from_path(model, path: str):
    """Navigate dotted path to get the QuantizedLinear module."""
    obj = model
    for part in path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def snapshot_weights(model, paths: list[str]) -> dict[str, mx.array]:
    """Save a copy of the weight tensors for later restoration."""
    snap = {}
    for p in paths:
        mod = get_weight_from_path(model, p)
        snap[p] = mx.array(mod.weight)
    return snap


def restore_weights(model, snapshot: dict[str, mx.array]):
    """Restore weight tensors from a snapshot."""
    updates = []
    for path, w in snapshot.items():
        updates.append((f"{path}.weight", w))
    model.load_weights(updates, strict=False)
    mx.eval(model.parameters())


def flip_random_bits(model, paths: list[str], n_flips: int,
                     rng: np.random.Generator):
    """Flip n_flips random bits across the given weight tensors."""
    # Build a list of (path, num_uint32s)
    tensor_info = []
    total_uint32s = 0
    for p in paths:
        mod = get_weight_from_path(model, p)
        count = mod.weight.size
        tensor_info.append((p, count))
        total_uint32s += count

    # Sample positions and bit indices
    positions = rng.integers(0, total_uint32s, size=n_flips)
    bit_indices = rng.integers(0, 32, size=n_flips)

    # Accumulate flip masks per tensor (in numpy for speed)
    masks = {}
    shapes = {}
    for p, count in tensor_info:
        mod = get_weight_from_path(model, p)
        shapes[p] = mod.weight.shape
        masks[p] = np.zeros(count, dtype=np.uint32)

    for pos, bit in zip(positions, bit_indices):
        cum = 0
        for p, count in tensor_info:
            if pos < cum + count:
                masks[p][pos - cum] ^= np.uint32(1 << bit)
                break
            cum += count

    # Apply XOR to model weights
    updates = []
    for p, count in tensor_info:
        if masks[p].any():
            mod = get_weight_from_path(model, p)
            mask_mx = mx.array(masks[p].reshape(shapes[p]))
            new_w = mod.weight ^ mask_mx
            updates.append((f"{p}.weight", new_w))

    if updates:
        model.load_weights(updates, strict=False)
        mx.eval(model.parameters())


def flip_scale_guided_bits(model, paths: list[str], n_flips: int,
                           rng: np.random.Generator, target: str = "medium"):
    """Flip bits in groups based on scale magnitude.
    target="medium": 25th-75th percentile scales
    target="high": top 25% scales
    """
    # Collect all scale magnitudes
    all_scales = []
    for p in paths:
        mod = get_weight_from_path(model, p)
        all_scales.append(np.array(mx.abs(mod.scales).reshape(-1)))
    all_scales = np.concatenate(all_scales)

    if target == "medium":
        lo, hi = np.percentile(all_scales, 25), np.percentile(all_scales, 75)
    else:  # high
        lo, hi = np.percentile(all_scales, 75), float(all_scales.max())

    # Build pool of (path, row, group_idx) for groups in target range
    group_pool = []
    for p in paths:
        mod = get_weight_from_path(model, p)
        scales = np.array(mx.abs(mod.scales))
        rows, groups = scales.shape
        mask = (scales >= lo) & (scales <= hi)
        rs, gs = np.where(mask)
        for r, g in zip(rs, gs):
            group_pool.append((p, int(r), int(g)))

    if not group_pool:
        print(f"  Warning: no groups in {target} range [{lo:.4f}, {hi:.4f}]")
        return

    # Sample groups and flip one bit in each
    chosen = rng.integers(0, len(group_pool), size=n_flips)

    # Build masks
    masks = {}
    shapes = {}
    for p in paths:
        mod = get_weight_from_path(model, p)
        shapes[p] = mod.weight.shape
        masks[p] = np.zeros(shapes[p], dtype=np.uint32)

    for idx in chosen:
        p, row, gidx = group_pool[idx]
        # Each group = 4 consecutive uint32s (4 * 32 = 128 bits)
        col = gidx * 4 + rng.integers(0, 4)
        bit = rng.integers(0, 32)
        masks[p][row, col] ^= np.uint32(1 << bit)

    updates = []
    for p in paths:
        if masks[p].any():
            mod = get_weight_from_path(model, p)
            mask_mx = mx.array(masks[p])
            updates.append((f"{p}.weight", mod.weight ^ mask_mx))

    if updates:
        model.load_weights(updates, strict=False)
        mx.eval(model.parameters())


def run_experiment():
    print("=" * 60)
    print("Bankai Experiment 1: XOR Patch Viability")
    print("=" * 60)

    print("\nLoading model...")
    t0 = time.time()
    model, tokenizer = load(MODEL_PATH)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("\nComputing baseline perplexity...")
    baseline = compute_perplexity(model, tokenizer, EVAL_TEXTS)
    print(f"  Baseline PPL: {baseline:.4f}")

    all_paths = get_mlp_weight_paths(model)
    mid_paths = get_mlp_weight_paths(model, layer_range=(16, 24))

    total_bits_all = sum(get_weight_from_path(model, p).weight.size * 32 for p in all_paths)
    total_bits_mid = sum(get_weight_from_path(model, p).weight.size * 32 for p in mid_paths)
    print(f"\n  All MLP layers: {len(all_paths)} tensors, {total_bits_all:,} bits")
    print(f"  Mid MLP (16-24): {len(mid_paths)} tensors, {total_bits_mid:,} bits")

    flip_counts = [100, 1_000, 10_000, 50_000, 100_000, 500_000]
    rng = np.random.default_rng(42)

    strategies = [
        ("random_all_mlp", "Random — all MLP layers", all_paths,
         lambda paths, n, r: flip_random_bits(model, paths, n, r)),
        ("random_mid_mlp", "Random — MLP layers 16-24", mid_paths,
         lambda paths, n, r: flip_random_bits(model, paths, n, r)),
        ("scale_medium", "Scale-guided (medium) — layers 16-24", mid_paths,
         lambda paths, n, r: flip_scale_guided_bits(model, paths, n, r, "medium")),
        ("scale_high", "Scale-guided (HIGH) — layers 16-24", mid_paths,
         lambda paths, n, r: flip_scale_guided_bits(model, paths, n, r, "high")),
    ]

    results = {
        "baseline_ppl": baseline,
        "flip_counts": flip_counts,
        "total_bits_all_mlp": total_bits_all,
        "total_bits_mid_mlp": total_bits_mid,
        "strategies": {},
    }

    for strat_id, strat_name, paths, flip_fn in strategies:
        print(f"\n--- {strat_name} ---")
        # Snapshot original weights for this set of paths
        snap = snapshot_weights(model, paths)
        ppls = []

        for n in flip_counts:
            # Apply flips
            flip_fn(paths, n, rng)

            ppl = compute_perplexity(model, tokenizer, EVAL_TEXTS)
            delta = ppl - baseline
            pct = (delta / baseline) * 100
            ppls.append(ppl)
            print(f"  {n:>8,} flips -> PPL: {ppl:>10.4f}  (Δ{delta:>+10.4f}, {pct:>+7.2f}%)")

            # Restore original weights
            restore_weights(model, snap)

        results["strategies"][strat_id] = {"name": strat_name, "perplexities": ppls}

    # Save
    out_path = Path("experiment_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Strategy':<42} {'PPL @ 10K flips':>16} {'Δ/1K flips':>12}")
    print("-" * 70)
    for strat_id, strat_name, _, _ in strategies:
        ppls = results["strategies"][strat_id]["perplexities"]
        ppl_10k = ppls[2]  # index 2 = 10K flips
        delta_per_1k = (ppl_10k - baseline) / 10
        print(f"  {strat_name:<40} {ppl_10k:>12.4f} {delta_per_1k:>+12.4f}")

    print(f"\n  Baseline PPL: {baseline:.4f}")
    print(f"  If scale-guided Δ >> random Δ, targeted XOR patches are viable.\n")

    return results


if __name__ == "__main__":
    run_experiment()
