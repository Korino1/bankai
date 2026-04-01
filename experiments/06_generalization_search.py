"""
Bankai Experiment 6: Generalization-Optimized Search
====================================================
Train on 60 probes (10 per category) with min-of-probes fitness.
Validate on 30 held-out probes (5 per category) never seen during search.

Tests whether more training signal + anti-overfitting fitness produces
patches that generalize beyond exact training prompts.
"""

from bankai.backends import get_backend
from bankai.patch import Patch, apply_patch, remove_patch
from bankai.probes import Probe, KNOWLEDGE_PROBES
from bankai.search import greedy_search

MODEL_PATH = "models/bonsai-8b-mlx"

# ── 10 training + 5 validation per category ──

POLY_DERIV_TRAIN = [
    Probe("d/dx [x^4 + 3x^2] =", " 4", " 0", "pd_train_0", "poly_deriv"),  # original
    Probe("d/dx [x^5 + 2x^3] =", " 5", " 0", "pd_train_1", "poly_deriv"),
    Probe("d/dx [x^3 + 7x] =", " 3", " 0", "pd_train_2", "poly_deriv"),
    Probe("d/dx [2x^4] =", " 8", " 0", "pd_train_3", "poly_deriv"),
    Probe("d/dx [x^6] =", " 6", " 0", "pd_train_4", "poly_deriv"),
    Probe("d/dx [x^2 + x] =", " 2", " 0", "pd_train_5", "poly_deriv"),
    Probe("d/dx [3x^3 + x^2] =", " 9", " 0", "pd_train_6", "poly_deriv"),
    Probe("d/dx [x^3 + 2x^2 + x] =", " 3", " 0", "pd_train_7", "poly_deriv"),
    Probe("The derivative of x^4 + 2x^3 is", " 4", " 0", "pd_train_8", "poly_deriv"),
    Probe("Differentiate x^5 + x with respect to x:", " 5", " 0", "pd_train_9", "poly_deriv"),
]
POLY_DERIV_VAL = [
    Probe("d/dx [x^7 + x] =", " 7", " 0", "pd_val_0", "poly_deriv"),
    Probe("d/dx [4x^3] =", " 12", " 0", "pd_val_1", "poly_deriv"),
    Probe("d/dx [x^4 + x^3 + x^2] =", " 4", " 0", "pd_val_2", "poly_deriv"),
    Probe("Find the derivative of 2x^5 + x:", " 10", " 0", "pd_val_3", "poly_deriv"),
    Probe("f(x) = x^6 + 3x, f'(x) =", " 6", " 0", "pd_val_4", "poly_deriv"),
]

SECOND_DERIV_TRAIN = [
    Probe("The second derivative of x^4 is 12x^", "2", "3", "sd_train_0", "second_deriv"),  # original
    Probe("The second derivative of x^5 is", " 20", " 0", "sd_train_1", "second_deriv"),
    Probe("The second derivative of x^3 is", " 6", " 0", "sd_train_2", "second_deriv"),
    Probe("d^2/dx^2 [x^4] =", " 12", " 0", "sd_train_3", "second_deriv"),
    Probe("d^2/dx^2 [x^5] =", " 20", " 0", "sd_train_4", "second_deriv"),
    Probe("d^2/dx^2 [x^3] =", " 6", " 0", "sd_train_5", "second_deriv"),
    Probe("f(x) = x^4, f''(x) =", " 12", " 0", "sd_train_6", "second_deriv"),
    Probe("The second derivative of x^6 is", " 30", " 0", "sd_train_7", "second_deriv"),
    Probe("The second derivative of 2x^4 is", " 24", " 0", "sd_train_8", "second_deriv"),
    Probe("Find the second derivative of x^5:", " 20", " 0", "sd_train_9", "second_deriv"),
]
SECOND_DERIV_VAL = [
    Probe("d^2/dx^2 [x^7] =", " 42", " 0", "sd_val_0", "second_deriv"),
    Probe("The second derivative of x^3 + x^2 is", " 6", " 0", "sd_val_1", "second_deriv"),
    Probe("f(x) = x^5, f''(x) =", " 20", " 0", "sd_val_2", "second_deriv"),
    Probe("The second derivative of 3x^3 is", " 18", " 0", "sd_val_3", "second_deriv"),
    Probe("What is f''(x) if f(x) = x^6?", " 30", " 0", "sd_val_4", "second_deriv"),
]

INTEGRAL_TRAIN = [
    Probe("The integral of x^2 dx = ", " 1", " 2", "int_train_0", "integral"),  # original
    Probe("The integral of x^3 dx =", " 1", " 2", "int_train_1", "integral"),
    Probe("The integral of x^4 dx =", " 1", " 2", "int_train_2", "integral"),
    Probe("The integral of x dx =", " 1", " 2", "int_train_3", "integral"),
    Probe("The integral of x^5 dx =", " 1", " 2", "int_train_4", "integral"),
    Probe("Evaluate the integral of x^2 dx:", " 1", " 2", "int_train_5", "integral"),
    Probe("Find the antiderivative of x^2:", " 1", " 2", "int_train_6", "integral"),
    Probe("The antiderivative of x^3 is", " 1", " 2", "int_train_7", "integral"),
    Probe("Integrate x^4 dx:", " 1", " 2", "int_train_8", "integral"),
    Probe("The indefinite integral of x^3 dx =", " 1", " 2", "int_train_9", "integral"),
]
INTEGRAL_VAL = [
    Probe("The integral of x^6 dx =", " 1", " 2", "int_val_0", "integral"),
    Probe("The integral of x^7 dx =", " 1", " 2", "int_val_1", "integral"),
    Probe("What is the integral of x^2 dx?", " 1", " 2", "int_val_2", "integral"),
    Probe("The antiderivative of x^4 is", " 1", " 2", "int_val_3", "integral"),
    Probe("The integral of x^2 with respect to x is", " 1", " 2", "int_val_4", "integral"),
]

PRIME_TRAIN = [
    Probe("Is 97 prime? Answer:", " Yes", " No", "pr_train_0", "prime"),  # original (no trailing space)
    Probe("Is 101 prime? Answer:", " Yes", " No", "pr_train_1", "prime"),
    Probe("Is 89 prime? Answer:", " Yes", " No", "pr_train_2", "prime"),
    Probe("Is 83 prime? Answer:", " Yes", " No", "pr_train_3", "prime"),
    Probe("Is 71 prime? Answer:", " Yes", " No", "pr_train_4", "prime"),
    Probe("Is 91 prime? Answer:", " No", " Yes", "pr_train_5", "prime"),  # not prime
    Probe("Is 87 prime? Answer:", " No", " Yes", "pr_train_6", "prime"),  # not prime
    Probe("Is 95 prime? Answer:", " No", " Yes", "pr_train_7", "prime"),  # not prime
    Probe("Is 67 prime? Answer:", " Yes", " No", "pr_train_8", "prime"),
    Probe("Is 99 prime? Answer:", " No", " Yes", "pr_train_9", "prime"),  # not prime
]
PRIME_VAL = [
    Probe("Is 107 prime? Answer:", " Yes", " No", "pr_val_0", "prime"),
    Probe("Is 113 prime? Answer:", " Yes", " No", "pr_val_1", "prime"),
    Probe("Is 53 prime? Answer:", " Yes", " No", "pr_val_2", "prime"),
    Probe("Is 77 prime? Answer:", " No", " Yes", "pr_val_3", "prime"),
    Probe("Is 119 prime? Answer:", " No", " Yes", "pr_val_4", "prime"),
]

TRIG_TRAIN = [
    Probe("sin(pi/6) =", " 1", " 0", "trig_train_0", "trig"),  # original (1/2)
    Probe("sin(pi/2) =", " 1", " 0", "trig_train_1", "trig"),
    Probe("cos(0) =", " 1", " 0", "trig_train_2", "trig"),
    Probe("sin(0) =", " 0", " 1", "trig_train_3", "trig"),
    Probe("cos(pi) =", " -", " 0", "trig_train_4", "trig"),
    Probe("tan(pi/4) =", " 1", " 0", "trig_train_5", "trig"),
    Probe("sin(pi) =", " 0", " 1", "trig_train_6", "trig"),
    Probe("cos(pi/2) =", " 0", " 1", "trig_train_7", "trig"),
    Probe("tan(0) =", " 0", " 1", "trig_train_8", "trig"),
    Probe("cos(2*pi) =", " 1", " 0", "trig_train_9", "trig"),
]
TRIG_VAL = [
    Probe("sin(pi/4) =", " 0", " 1", "trig_val_0", "trig"),
    Probe("cos(pi/3) =", " 0", " 1", "trig_val_1", "trig"),
    Probe("sin(pi/3) =", " 0", " 1", "trig_val_2", "trig"),
    Probe("cos(pi/6) =", " 0", " 1", "trig_val_3", "trig"),
    Probe("sin(3*pi/2) =", " -", " 0", "trig_val_4", "trig"),
]

EXP_DERIV_TRAIN = [
    Probe("d/dx [e^(2x)] = ", " 2", " 6", "ed_train_0", "exp_deriv"),  # original
    Probe("d/dx [e^(3x)] =", " 3", " 9", "ed_train_1", "exp_deriv"),
    Probe("d/dx [e^(4x)] =", " 4", " 16", "ed_train_2", "exp_deriv"),
    Probe("d/dx [e^x] =", " e", " x", "ed_train_3", "exp_deriv"),
    Probe("d/dx [e^(-x)] =", " -", " e", "ed_train_4", "exp_deriv"),
    Probe("The derivative of e^(2x) is", " 2", " 4", "ed_train_5", "exp_deriv"),
    Probe("The derivative of e^(3x) is", " 3", " 9", "ed_train_6", "exp_deriv"),
    Probe("f(x) = e^(2x), f'(x) =", " 2", " 4", "ed_train_7", "exp_deriv"),
    Probe("Differentiate e^(2x):", " 2", " 4", "ed_train_8", "exp_deriv"),
    Probe("The derivative of e^x is", " e", " x", "ed_train_9", "exp_deriv"),
]
EXP_DERIV_VAL = [
    Probe("d/dx [e^(5x)] =", " 5", " 25", "ed_val_0", "exp_deriv"),
    Probe("f(x) = e^(3x), f'(x) =", " 3", " 9", "ed_val_1", "exp_deriv"),
    Probe("d/dx [e^(-2x)] =", " -", " 4", "ed_val_2", "exp_deriv"),
    Probe("d/dx [2*e^x] =", " 2", " x", "ed_val_3", "exp_deriv"),
    Probe("d/dx [e^(x^2)] =", " 2", " e", "ed_val_4", "exp_deriv"),
]

ALL_TRAIN = (
    POLY_DERIV_TRAIN + SECOND_DERIV_TRAIN + INTEGRAL_TRAIN +
    PRIME_TRAIN + TRIG_TRAIN + EXP_DERIV_TRAIN
)

ALL_VAL = (
    POLY_DERIV_VAL + SECOND_DERIV_VAL + INTEGRAL_VAL +
    PRIME_VAL + TRIG_VAL + EXP_DERIV_VAL
)

VAL_BY_CATEGORY = [
    ("poly_deriv", POLY_DERIV_VAL),
    ("second_deriv", SECOND_DERIV_VAL),
    ("integral", INTEGRAL_VAL),
    ("prime", PRIME_VAL),
    ("trig", TRIG_VAL),
    ("exp_deriv", EXP_DERIV_VAL),
]


def measure_gaps(backend, probes):
    gaps = {}
    for p in probes:
        tokens = backend.encode(p.prompt)
        c_id = backend.encode_token(p.correct_token)
        w_id = backend.encode_token(p.wrong_token)
        gaps[p.name] = backend.logit_gap(tokens, c_id, w_id)
    return gaps


def sign_flip_analysis(baseline, patched, probes):
    fixed, broke, stayed_right, stayed_wrong = [], [], [], []
    for p in probes:
        b = baseline[p.name]
        a = patched[p.name]
        if b <= 0 and a > 0:
            fixed.append(p.name)
        elif b > 0 and a <= 0:
            broke.append(p.name)
        elif b > 0:
            stayed_right.append(p.name)
        else:
            stayed_wrong.append(p.name)
    return fixed, broke, stayed_right, stayed_wrong


def main():
    print("=" * 70)
    print("Bankai Experiment 6: Generalization-Optimized Search")
    print(f"  Training: {len(ALL_TRAIN)} probes (10 per category)")
    print(f"  Validation: {len(ALL_VAL)} probes (5 per category, held out)")
    print(f"  Fitness: mean (regularized by probe diversity)")
    print("=" * 70)

    backend = get_backend("mlx")
    backend.load(MODEL_PATH)

    # Search
    patch = greedy_search(
        backend,
        target_probes=ALL_TRAIN,
        control_probes=KNOWLEDGE_PROBES,
        search_layers=[1, 2, 3, 4, 34],
        search_projs=["gate_proj", "up_proj"],
        max_iters=300,
        control_penalty=2.0,
        fitness_mode="mean",
        seed=42,
        patch_name="calculus_generalized_v1",
        patch_description="Calculus patch trained on 60 probes with min-of-probes fitness",
    )

    patch.save("patches/calculus_generalized_v1.json")
    print(f"\nPatch saved: {len(patch.flips)} flips, {patch.size_bytes} bytes")

    # ── Evaluate on TRAINING probes ──
    # NOTE: greedy_search leaves accepted flips applied, so remove first
    # to get a clean baseline, then re-apply for patched measurement.
    print(f"\n{'='*70}")
    print("TRAINING SET EVALUATION")
    print(f"{'='*70}")

    remove_patch(backend, patch)
    baseline_train = measure_gaps(backend, ALL_TRAIN)
    apply_patch(backend, patch)
    patched_train = measure_gaps(backend, ALL_TRAIN)

    fixed, broke, right, wrong = sign_flip_analysis(baseline_train, patched_train, ALL_TRAIN)
    print(f"  Fixed (wrong→right): {len(fixed)}")
    print(f"  Broke (right→wrong): {len(broke)}")
    print(f"  Stayed right:        {len(right)}")
    print(f"  Stayed wrong:        {len(wrong)}")
    print(f"  Accuracy: {len(right)+len(broke)}/{len(ALL_TRAIN)} → {len(right)+len(fixed)}/{len(ALL_TRAIN)}")

    # ── Evaluate on VALIDATION probes (never seen during search) ──
    print(f"\n{'='*70}")
    print("VALIDATION SET EVALUATION (never seen during search)")
    print(f"{'='*70}")

    # Model still has patch applied from above
    patched_val = measure_gaps(backend, ALL_VAL)
    remove_patch(backend, patch)
    baseline_val = measure_gaps(backend, ALL_VAL)

    fixed_v, broke_v, right_v, wrong_v = sign_flip_analysis(baseline_val, patched_val, ALL_VAL)
    print(f"  Fixed (wrong→right): {len(fixed_v)}")
    print(f"  Broke (right→wrong): {len(broke_v)}")
    print(f"  Stayed right:        {len(right_v)}")
    print(f"  Stayed wrong:        {len(wrong_v)}")
    print(f"  Accuracy: {len(right_v)+len(broke_v)}/{len(ALL_VAL)} → {len(right_v)+len(fixed_v)}/{len(ALL_VAL)}")

    # Per-category breakdown
    print(f"\n  {'Category':<20} {'Fixed':>6} {'Broke':>6} {'Right':>6} {'Wrong':>6}")
    print(f"  {'-'*46}")
    for cat_name, val_probes in VAL_BY_CATEGORY:
        f, b, r, w = sign_flip_analysis(baseline_val, patched_val, val_probes)
        print(f"  {cat_name:<20} {len(f):>6} {len(b):>6} {len(r):>6} {len(w):>6}")

    if fixed_v:
        print(f"\n  Fixed probes:")
        for name in fixed_v:
            print(f"    {name}: {baseline_val[name]:>+.2f} → {patched_val[name]:>+.2f}")
    if broke_v:
        print(f"\n  Broke probes:")
        for name in broke_v:
            print(f"    {name}: {baseline_val[name]:>+.2f} → {patched_val[name]:>+.2f}")


if __name__ == "__main__":
    main()
