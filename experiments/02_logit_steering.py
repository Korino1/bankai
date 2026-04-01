"""
Bankai Experiment 2: Logit Steering via Bit Flips
==================================================
Can targeted bit flips shift the model's probability distribution
in a specific, measurable direction?

We measure logit gaps on carefully chosen prompts where we know
what the model "should" say, then flip bits and see if we can
move the distribution.

Key insight from Exp 1: random flips barely move perplexity even
at 500K flips. So we try CORRELATED flips — entire rows (neurons),
entire groups, or all bits in a single layer — to see if structured
modifications produce structured behavioral changes.
"""

import time
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load

MODEL_PATH = "models/bonsai-8b-mlx"

# Probes: (prompt, token_a, token_b, description)
# We measure P(token_a) vs P(token_b) — the "logit gap"
PROBES = [
    ("The capital of France is", " Paris", " London", "geography_france"),
    ("The capital of Japan is", " Tokyo", " Beijing", "geography_japan"),
    ("2 + 2 =", " 4", " 5", "arithmetic"),
    ("The color of the sky is", " blue", " red", "common_knowledge"),
    ("def hello():\n    print(\"Hello", " World", " Goodbye", "code_completion"),
    ("In Python, to open a file you use the", " open", " close", "python_api"),
    ("The chemical formula for water is H", "2", "3", "chemistry"),
    ("Einstein is famous for the theory of", " relativity", " evolution", "physics"),
]


def get_logit_gaps(model, tokenizer, probes=PROBES):
    """For each probe, return logit(token_a) - logit(token_b)."""
    gaps = {}
    for prompt, tok_a, tok_b, name in probes:
        tokens = mx.array(tokenizer.encode(prompt))
        logits = model(tokens[None, :])
        last = logits[0, -1, :]
        mx.eval(last)

        id_a = tokenizer.encode(tok_a)[-1]
        id_b = tokenizer.encode(tok_b)[-1]
        gap = last[id_a].item() - last[id_b].item()
        prob_a = mx.softmax(last)[id_a].item()
        gaps[name] = {"gap": gap, "prob_correct": prob_a}
    return gaps


def get_module(model, path: str):
    obj = model
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def snapshot_layer_weights(model, layer_idx: int):
    """Snapshot all quantized weight tensors in a layer."""
    snap = {}
    prefix = f"model.layers.{layer_idx}"
    projections = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]
    for proj in projections:
        path = f"{prefix}.{proj}"
        mod = get_module(model, path)
        snap[f"{path}.weight"] = mx.array(mod.weight)
    return snap


def restore_snapshot(model, snap):
    model.load_weights(list(snap.items()), strict=False)
    mx.eval(model.parameters())


def flip_entire_rows(model, layer_idx: int, proj: str, row_indices: list[int]):
    """Flip ALL bits in specific rows of a weight matrix.
    A row = one output neuron. This is a maximally correlated flip."""
    path = f"model.layers.{layer_idx}.mlp.{proj}"
    mod = get_module(model, path)
    w = mod.weight  # (out_features, packed_cols) uint32

    # XOR with all-ones for selected rows = flip every bit in those rows
    mask = mx.zeros_like(w)
    ones_row = mx.full((w.shape[1],), 0xFFFFFFFF, dtype=mx.uint32)
    for r in row_indices:
        mask = mask.at[r].add(ones_row)

    new_w = w ^ mask
    model.load_weights([(f"{path}.weight", new_w)], strict=False)
    mx.eval(model.parameters())


def flip_entire_layer(model, layer_idx: int, component: str = "mlp"):
    """Flip ALL bits in all projections of a layer component."""
    if component == "mlp":
        projs = ["gate_proj", "up_proj", "down_proj"]
    else:
        projs = ["q_proj", "k_proj", "v_proj", "o_proj"]

    updates = []
    for proj in projs:
        path = f"model.layers.{layer_idx}.{component}.{proj}"
        mod = get_module(model, path)
        w = mod.weight
        mask = mx.full(w.shape, 0xFFFFFFFF, dtype=mx.uint32)
        updates.append((f"{path}.weight", w ^ mask))

    model.load_weights(updates, strict=False)
    mx.eval(model.parameters())


def flip_random_rows(model, layer_idx: int, proj: str, n_rows: int, rng):
    """Flip all bits in N random rows of a projection."""
    path = f"model.layers.{layer_idx}.mlp.{proj}"
    mod = get_module(model, path)
    total_rows = mod.weight.shape[0]
    rows = rng.choice(total_rows, size=min(n_rows, total_rows), replace=False).tolist()
    flip_entire_rows(model, layer_idx, proj, rows)
    return len(rows)


def flip_high_scale_rows(model, layer_idx: int, proj: str, n_rows: int):
    """Flip all bits in the N rows whose groups have the highest avg scale."""
    path = f"model.layers.{layer_idx}.mlp.{proj}"
    mod = get_module(model, path)
    # Average scale magnitude per row
    row_scales = mx.mean(mx.abs(mod.scales), axis=1)
    mx.eval(row_scales)
    top_rows = mx.argsort(row_scales)[-n_rows:].tolist()
    flip_entire_rows(model, layer_idx, proj, top_rows)
    return top_rows


def run_experiment():
    print("=" * 65)
    print("Bankai Experiment 2: Logit Steering via Structured Bit Flips")
    print("=" * 65)

    model, tokenizer = load(MODEL_PATH)
    rng = np.random.default_rng(42)

    # Baseline
    print("\n[Baseline]")
    baseline = get_logit_gaps(model, tokenizer)
    for name, data in baseline.items():
        print(f"  {name:<25} gap={data['gap']:>+8.3f}  P(correct)={data['prob_correct']:.4f}")

    results = {"baseline": baseline, "experiments": []}

    # ── Experiment A: Flip entire MLP in one layer ──
    # Test each layer to find which layers matter most
    print("\n[A] Flip entire MLP per layer — which layers are load-bearing?")
    print(f"  {'Layer':>5} {'Avg |Δgap|':>12} {'Max |Δgap|':>12} {'Broken probes':>14}")
    print("  " + "-" * 47)

    layer_impacts = []
    for layer_idx in range(36):
        snap = snapshot_layer_weights(model, layer_idx)
        flip_entire_layer(model, layer_idx, "mlp")

        flipped = get_logit_gaps(model, tokenizer)
        restore_snapshot(model, snap)

        deltas = [abs(flipped[n]["gap"] - baseline[n]["gap"]) for n in baseline]
        avg_delta = np.mean(deltas)
        max_delta = np.max(deltas)
        broken = sum(1 for n in baseline
                     if np.sign(flipped[n]["gap"]) != np.sign(baseline[n]["gap"]))

        layer_impacts.append((layer_idx, avg_delta, max_delta, broken))
        flag = " <<<" if avg_delta > 1.0 else ""
        print(f"  {layer_idx:>5} {avg_delta:>12.3f} {max_delta:>12.3f} {broken:>14}{flag}")

    results["experiments"].append({
        "name": "flip_entire_mlp_per_layer",
        "data": [(i, float(a), float(m), b) for i, a, m, b in layer_impacts]
    })

    # Find the most impactful layers
    layer_impacts.sort(key=lambda x: x[1], reverse=True)
    top_layers = [x[0] for x in layer_impacts[:5]]
    print(f"\n  Most impactful layers: {top_layers}")

    # ── Experiment B: Row-level flips in the most impactful layer ──
    best_layer = top_layers[0]
    print(f"\n[B] Row-level flips in layer {best_layer} gate_proj — how many rows to move the needle?")
    print(f"  {'N rows':>8} {'Avg |Δgap|':>12} {'Broken':>8}")
    print("  " + "-" * 32)

    for n_rows in [1, 4, 16, 64, 256, 1024]:
        snap = snapshot_layer_weights(model, best_layer)
        flip_random_rows(model, best_layer, "gate_proj", n_rows, rng)

        flipped = get_logit_gaps(model, tokenizer)
        restore_snapshot(model, snap)

        deltas = [abs(flipped[n]["gap"] - baseline[n]["gap"]) for n in baseline]
        avg_d = np.mean(deltas)
        broken = sum(1 for n in baseline
                     if np.sign(flipped[n]["gap"]) != np.sign(baseline[n]["gap"]))
        print(f"  {n_rows:>8} {avg_d:>12.3f} {broken:>8}")

    # ── Experiment C: High-scale rows vs random rows ──
    print(f"\n[C] High-scale rows vs random rows (64 rows in layer {best_layer} gate_proj)")

    snap = snapshot_layer_weights(model, best_layer)

    # High-scale
    flip_high_scale_rows(model, best_layer, "gate_proj", 64)
    flipped_high = get_logit_gaps(model, tokenizer)
    restore_snapshot(model, snap)

    # Random
    flip_random_rows(model, best_layer, "gate_proj", 64, rng)
    flipped_rand = get_logit_gaps(model, tokenizer)
    restore_snapshot(model, snap)

    print(f"  {'Probe':<25} {'Baseline':>9} {'High-Δ':>9} {'Rand-Δ':>9} {'High>Rand?':>11}")
    print("  " + "-" * 66)
    for name in baseline:
        bg = baseline[name]["gap"]
        hd = abs(flipped_high[name]["gap"] - bg)
        rd = abs(flipped_rand[name]["gap"] - bg)
        winner = "YES" if hd > rd else "no"
        print(f"  {name:<25} {bg:>+9.3f} {hd:>9.3f} {rd:>9.3f} {winner:>11}")

    high_avg = np.mean([abs(flipped_high[n]["gap"] - baseline[n]["gap"]) for n in baseline])
    rand_avg = np.mean([abs(flipped_rand[n]["gap"] - baseline[n]["gap"]) for n in baseline])
    print(f"\n  Average |Δgap|:  high-scale={high_avg:.3f}  random={rand_avg:.3f}  ratio={high_avg/max(rand_avg,1e-6):.2f}x")

    # ── Experiment D: Layer-specific effects on different domains ──
    print(f"\n[D] Do different layers affect different probes? (top 3 layers, full MLP flip)")
    print(f"  {'Probe':<25}", end="")
    for l in top_layers[:3]:
        print(f" {'L'+str(l)+' Δgap':>10}", end="")
    print()
    print("  " + "-" * 55)

    for l in top_layers[:3]:
        snap = snapshot_layer_weights(model, l)
        flip_entire_layer(model, l, "mlp")
        flipped = get_logit_gaps(model, tokenizer)
        restore_snapshot(model, snap)
        results[f"layer_{l}_flip"] = flipped

    # Re-run to print (we already have the data from the loop above, but let's be clean)
    layer_flipped = {}
    for l in top_layers[:3]:
        snap = snapshot_layer_weights(model, l)
        flip_entire_layer(model, l, "mlp")
        layer_flipped[l] = get_logit_gaps(model, tokenizer)
        restore_snapshot(model, snap)

    for name in baseline:
        print(f"  {name:<25}", end="")
        for l in top_layers[:3]:
            delta = layer_flipped[l][name]["gap"] - baseline[name]["gap"]
            print(f" {delta:>+10.3f}", end="")
        print()

    # Save
    out = Path("experiment2_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    run_experiment()
