"""
Bankai Experiment 4: Calculus Patch Search
==========================================
Target: fix verified calculus/advanced math failures on Bonsai 8B.
Control: preserve basic knowledge probes.
"""

import time
from bankai.backends import get_backend
from bankai.probes import Probe, KNOWLEDGE_PROBES, measure_probes
from bankai.search import greedy_search
from bankai.patch import apply_patch, remove_patch

MODEL_PATH = "models/bonsai-8b-mlx"

CALCULUS_PROBES = [
    Probe("The integral of x^2 dx = ", " 1", " 2", "poly_integral", "calculus"),
    Probe("The second derivative of x^4 is 12x^", "2", "3", "second_deriv", "calculus"),
    Probe("Is 97 prime? Answer: ", " Yes", " No", "prime_97", "math"),
    Probe("sin(pi/6) = ", " 1", " 0", "sin_pi6", "trig"),
    Probe("d/dx [x^4 + 3x^2] =", " 4", " 0", "poly_deriv", "calculus"),
    Probe("d/dx [e^(2x)] = ", " 2", " 6", "exp_deriv", "calculus"),
]

def main():
    print("Loading model...")
    backend = get_backend("mlx")
    backend.load(MODEL_PATH)

    print("\nBaseline (calculus probes):")
    baseline = measure_probes(backend, CALCULUS_PROBES)
    for name, gap in baseline.items():
        status = "✓" if gap > 0 else "✗"
        print(f"  {name:<20} gap={gap:>+8.3f} {status}")

    print("\nSearching for calculus patch (300 iterations)...")
    patch = greedy_search(
        backend,
        target_probes=CALCULUS_PROBES,
        control_probes=KNOWLEDGE_PROBES,
        search_layers=[1, 2, 3, 4, 34],
        search_projs=["gate_proj", "up_proj"],
        max_iters=300,
        control_penalty=2.0,
        seed=123,
        patch_name="calculus_v1",
        patch_description="Improves calculus and advanced math reasoning",
    )

    print("\n\nAfter patch (calculus probes):")
    patched = measure_probes(backend, CALCULUS_PROBES)
    for name in baseline:
        b = baseline[name]
        p = patched[name]
        delta = p - b
        print(f"  {name:<20} {b:>+8.3f} -> {p:>+8.3f}  (Δ{delta:>+8.3f})")

    print("\nControl probes (post-search):")
    ctrl_patched = measure_probes(backend, KNOWLEDGE_PROBES)
    for name, gap in ctrl_patched.items():
        status = "✓" if gap > 0 else "✗"
        print(f"  {name:<20} gap={gap:>+8.3f} {status}")

    # Save
    patch.save("patches/calculus_v1.json")
    print(f"\nPatch saved: {len(patch.flips)} flips, {patch.size_bytes} bytes")

    # Generation test
    test_prompts = [
        "The integral of x^2 dx = ",
        "Is 97 prime? Answer: ",
        "The second derivative of x^4 is",
        "sin(pi/6) =",
        "d/dx [x^4 + 3x^2] =",
    ]

    remove_patch(backend, patch)
    print("\n=== WITHOUT PATCH ===")
    for p in test_prompts:
        r = backend.generate(p, max_tokens=15)
        short = r.strip().split('\n')[0][:50]
        print(f"  {p:<45} {short}")

    apply_patch(backend, patch)
    print("\n=== WITH PATCH ===")
    for p in test_prompts:
        r = backend.generate(p, max_tokens=15)
        short = r.strip().split('\n')[0][:50]
        print(f"  {p:<45} {short}")


if __name__ == "__main__":
    main()
