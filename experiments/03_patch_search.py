"""
Bankai Experiment 3: Greedy Patch Search
========================================
Find the minimal set of row-level bit flips that shifts model behavior
in a targeted direction while preserving other capabilities.

Strategy: greedy hill climbing over rows in high-impact layers.
Each step flips one row, keeps it if fitness improves, reverts if not.
The accumulated flips at the end = the XOR patch.
"""

import time
import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

MODEL_PATH = "models/bonsai-8b-mlx"

# ── Probe definitions ──
# Target: we want to IMPROVE these (push correct answer probability up)
TARGET_PROBES = [
    ("1 + 1 =", " 2", " 3", "math_1"),
    ("2 + 2 =", " 4", " 5", "math_2"),
    ("7 * 8 =", " 56", " 54", "math_3"),
    ("The square root of 144 is", " 12", " 14", "math_4"),
    ("If x = 3, then x^2 =", " 9", " 8", "math_5"),
    ("100 / 4 =", " 25", " 20", "math_6"),
]

# Control: we want to PRESERVE these (don't degrade)
CONTROL_PROBES = [
    ("The capital of France is", " Paris", " London", "geo_1"),
    ("The capital of Japan is", " Tokyo", " Beijing", "geo_2"),
    ("The color of the sky is", " blue", " red", "knowledge_1"),
    ("Einstein is famous for the theory of", " relativity", " evolution", "knowledge_2"),
]

# Layers to search (from experiment 2: most impactful)
SEARCH_LAYERS = [34, 3, 1, 2, 4]
# Projections to search
SEARCH_PROJS = ["gate_proj", "up_proj"]
# How many candidate rows to try per iteration
CANDIDATES_PER_STEP = 1
# Total search iterations
MAX_ITERS = 200
# Penalty weight for control degradation
CONTROL_PENALTY = 2.0


def get_module(model, path: str):
    obj = model
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def measure_probes(model, tokenizer, probes):
    """Return dict of {name: logit_gap} for each probe."""
    gaps = {}
    for prompt, tok_a, tok_b, name in probes:
        tokens = mx.array(tokenizer.encode(prompt))
        logits = model(tokens[None, :])
        last = logits[0, -1, :]
        mx.eval(last)
        id_a = tokenizer.encode(tok_a)[-1]
        id_b = tokenizer.encode(tok_b)[-1]
        gaps[name] = last[id_a].item() - last[id_b].item()
    return gaps


def compute_fitness(target_gaps, control_gaps, target_baseline, control_baseline):
    """Fitness = avg target improvement - penalty * avg control degradation."""
    # Target: higher gap = better (correct answer more likely)
    target_improvement = np.mean([
        target_gaps[n] - target_baseline[n] for n in target_baseline
    ])
    # Control: we penalize if gaps decrease (correct answer becomes less likely)
    control_degradation = np.mean([
        max(0, control_baseline[n] - control_gaps[n]) for n in control_baseline
    ])
    return target_improvement - CONTROL_PENALTY * control_degradation


def build_search_candidates(model, rng):
    """Build list of (layer, proj, row_idx) candidates, biased toward high-scale rows."""
    candidates = []
    for layer_idx in SEARCH_LAYERS:
        for proj in SEARCH_PROJS:
            path = f"model.layers.{layer_idx}.mlp.{proj}"
            mod = get_module(model, path)
            n_rows = mod.weight.shape[0]
            # Get row-level scale magnitudes
            row_scales = np.array(mx.mean(mx.abs(mod.scales), axis=1))
            # Sample proportional to scale magnitude (high-scale rows more likely)
            probs = row_scales / row_scales.sum()
            for row in range(n_rows):
                candidates.append((layer_idx, proj, row, probs[row]))
    return candidates


def apply_row_flip(model, layer_idx: int, proj: str, row_idx: int):
    """Flip all bits in one row. Returns the update tuple for later reversal."""
    path = f"model.layers.{layer_idx}.mlp.{proj}"
    mod = get_module(model, path)
    w = mod.weight
    mask = mx.zeros_like(w)
    ones_row = mx.full((w.shape[1],), 0xFFFFFFFF, dtype=mx.uint32)
    mask = mask.at[row_idx].add(ones_row)
    new_w = w ^ mask
    model.load_weights([(f"{path}.weight", new_w)], strict=False)
    mx.eval(model.parameters())


def save_patch(accepted_flips: list[tuple[int, str, int]], output_path: str):
    """Save patch as a compact JSON + binary format."""
    patch = {
        "version": 1,
        "type": "row_flip",
        "description": "Math reasoning improvement patch",
        "base_model": "prism-ml/Bonsai-8B-mlx-1bit",
        "flips": [
            {"layer": l, "proj": p, "row": r}
            for l, p, r in accepted_flips
        ],
        "n_flips": len(accepted_flips),
        # Each row flip = 128 uint32s = 4096 bits
        "bits_flipped": len(accepted_flips) * 4096,
        "patch_size_bytes": len(accepted_flips) * 12,  # 3 ints per flip
    }
    with open(output_path, "w") as f:
        json.dump(patch, f, indent=2)
    return patch


def run_search():
    print("=" * 65)
    print("Bankai Experiment 3: Greedy Patch Search")
    print("=" * 65)

    model, tokenizer = load(MODEL_PATH)
    rng = np.random.default_rng(42)

    # ── Baselines ──
    print("\n[Baseline measurements]")
    target_baseline = measure_probes(model, tokenizer, TARGET_PROBES)
    control_baseline = measure_probes(model, tokenizer, CONTROL_PROBES)

    print("  Target probes (math):")
    for name, gap in target_baseline.items():
        status = "✓" if gap > 0 else "✗"
        print(f"    {name}: gap={gap:>+8.3f} {status}")
    print("  Control probes (knowledge):")
    for name, gap in control_baseline.items():
        status = "✓" if gap > 0 else "✗"
        print(f"    {name}: gap={gap:>+8.3f} {status}")

    baseline_fitness = compute_fitness(
        target_baseline, control_baseline, target_baseline, control_baseline
    )
    print(f"\n  Baseline fitness: {baseline_fitness:.4f}")

    # ── Build candidate pool ──
    candidates = build_search_candidates(model, rng)
    candidate_weights = np.array([c[3] for c in candidates])
    candidate_weights /= candidate_weights.sum()
    print(f"  Candidate pool: {len(candidates)} rows across {len(SEARCH_LAYERS)} layers")

    # ── Greedy search ──
    print(f"\n[Searching — {MAX_ITERS} iterations]")
    accepted_flips = []
    current_fitness = baseline_fitness
    tried = set()
    accept_count = 0

    t0 = time.time()
    for step in range(MAX_ITERS):
        # Sample a candidate (weighted by scale)
        while True:
            idx = rng.choice(len(candidates), p=candidate_weights)
            layer_idx, proj, row_idx, _ = candidates[idx]
            key = (layer_idx, proj, row_idx)
            if key not in tried:
                tried.add(key)
                break
            if len(tried) >= len(candidates):
                print("  Exhausted all candidates!")
                break

        # Apply flip
        apply_row_flip(model, layer_idx, proj, row_idx)

        # Measure
        target_gaps = measure_probes(model, tokenizer, TARGET_PROBES)
        control_gaps = measure_probes(model, tokenizer, CONTROL_PROBES)
        fitness = compute_fitness(target_gaps, control_gaps, target_baseline, control_baseline)

        if fitness > current_fitness:
            # Accept
            accepted_flips.append(key)
            current_fitness = fitness
            accept_count += 1
            marker = "++ACCEPT++"
        else:
            # Revert
            apply_row_flip(model, layer_idx, proj, row_idx)  # XOR again = undo
            marker = "  revert"

        if (step + 1) % 10 == 0 or "ACCEPT" in marker:
            elapsed = time.time() - t0
            rate = (step + 1) / elapsed
            print(f"  Step {step+1:>4}/{MAX_ITERS}  "
                  f"fitness={fitness:>+8.4f}  best={current_fitness:>+8.4f}  "
                  f"accepted={accept_count}  [{rate:.1f} it/s]  "
                  f"L{layer_idx}.{proj}[{row_idx}] {marker}")

    elapsed = time.time() - t0
    print(f"\n  Search complete in {elapsed:.1f}s ({MAX_ITERS/elapsed:.1f} it/s)")
    print(f"  Accepted {accept_count} of {MAX_ITERS} candidates")

    # ── Final evaluation ──
    print("\n[Final measurements with patch applied]")
    final_target = measure_probes(model, tokenizer, TARGET_PROBES)
    final_control = measure_probes(model, tokenizer, CONTROL_PROBES)

    print("  Target probes (math):")
    for name in target_baseline:
        before = target_baseline[name]
        after = final_target[name]
        delta = after - before
        print(f"    {name}: {before:>+8.3f} -> {after:>+8.3f}  (Δ{delta:>+8.3f})")

    print("  Control probes (knowledge):")
    for name in control_baseline:
        before = control_baseline[name]
        after = final_control[name]
        delta = after - before
        print(f"    {name}: {before:>+8.3f} -> {after:>+8.3f}  (Δ{delta:>+8.3f})")

    final_fitness = compute_fitness(final_target, final_control, target_baseline, control_baseline)
    print(f"\n  Fitness: {baseline_fitness:.4f} -> {final_fitness:.4f}  (Δ{final_fitness - baseline_fitness:+.4f})")

    # ── Save patch ──
    patch = save_patch(accepted_flips, "patch_math_v1.json")
    bits_flipped = patch["bits_flipped"]
    total_bits = 5_435_817_984  # from experiment 1
    print(f"\n[Patch saved to patch_math_v1.json]")
    print(f"  Rows flipped: {len(accepted_flips)}")
    print(f"  Bits flipped: {bits_flipped:,} ({bits_flipped/total_bits*100:.6f}% of all MLP bits)")
    print(f"  Patch metadata size: {patch['patch_size_bytes']} bytes")

    # ── Verify reversibility ──
    print("\n[Verify: removing patch restores baseline]")
    for layer_idx, proj, row_idx in accepted_flips:
        apply_row_flip(model, layer_idx, proj, row_idx)
    restored_target = measure_probes(model, tokenizer, TARGET_PROBES)
    max_drift = max(abs(restored_target[n] - target_baseline[n]) for n in target_baseline)
    print(f"  Max drift after patch removal: {max_drift:.6f}")
    print(f"  Reversibility: {'VERIFIED' if max_drift < 0.01 else 'FAILED'}")


if __name__ == "__main__":
    run_search()
