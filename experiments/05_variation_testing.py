"""
Bankai Experiment 5: Variation Testing
======================================
Does the calculus patch generalize beyond the exact prompts it was optimized for?

For each of the 6 probe categories, we test 15 novel variations that the patch
never saw during search. This answers: did we shift a capability region, or
memorize specific prompts?
"""

from bankai.backends import get_backend
from bankai.patch import Patch, apply_patch, remove_patch
from bankai.probes import Probe


MODEL_PATH = "models/bonsai-8b-mlx"
PATCH_PATH = "patches/calculus_v1.json"


# ── Variations: 15 per category, none seen during search ──

POLY_DERIV_VARIATIONS = [
    # Original: d/dx [x^4 + 3x^2] = , correct " 4", wrong " 0"
    # These test: does the model now handle polynomial differentiation generally?
    Probe("d/dx [x^5 + 2x^3] =", " 5", " 0", "poly_d_1", "poly_deriv"),
    Probe("d/dx [x^3 + 7x] =", " 3", " 0", "poly_d_2", "poly_deriv"),
    Probe("d/dx [2x^4] =", " 8", " 0", "poly_d_3", "poly_deriv"),
    Probe("d/dx [x^6] =", " 6", " 0", "poly_d_4", "poly_deriv"),
    Probe("d/dx [x^2 + x] =", " 2", " 0", "poly_d_5", "poly_deriv"),
    Probe("d/dx [3x^3 + x^2] =", " 9", " 0", "poly_d_6", "poly_deriv"),
    Probe("d/dx [x^4 + x] =", " 4", " 0", "poly_d_7", "poly_deriv"),
    Probe("d/dx [5x^2] =", " 10", " 0", "poly_d_8", "poly_deriv"),
    Probe("d/dx [x^3 + 2x^2 + x] =", " 3", " 0", "poly_d_9", "poly_deriv"),
    Probe("d/dx [x^7] =", " 7", " 0", "poly_d_10", "poly_deriv"),
    # Rephrased prompts (same math, different wording)
    Probe("Find the derivative of x^4 + 3x^2. Answer:", " 4", " 0", "poly_d_rephrase_1", "poly_deriv"),
    Probe("Differentiate x^5 + x with respect to x:", " 5", " 0", "poly_d_rephrase_2", "poly_deriv"),
    Probe("What is the derivative of x^3 + 4x? Answer:", " 3", " 0", "poly_d_rephrase_3", "poly_deriv"),
    Probe("The derivative of x^4 + 2x^3 is", " 4", " 0", "poly_d_rephrase_4", "poly_deriv"),
    Probe("f(x) = x^5 + x^2, f'(x) =", " 5", " 0", "poly_d_rephrase_5", "poly_deriv"),
]

SECOND_DERIV_VARIATIONS = [
    # Original: "The second derivative of x^4 is 12x^", correct "2", wrong "3"
    Probe("The second derivative of x^5 is", " 20", " 0", "sec_d_1", "second_deriv"),
    Probe("The second derivative of x^3 is", " 6", " 0", "sec_d_2", "second_deriv"),
    Probe("The second derivative of x^6 is", " 30", " 0", "sec_d_3", "second_deriv"),
    Probe("The second derivative of 2x^4 is", " 24", " 0", "sec_d_4", "second_deriv"),
    Probe("The second derivative of x^3 + x^2 is", " 6", " 0", "sec_d_5", "second_deriv"),
    Probe("d^2/dx^2 [x^4] =", " 12", " 0", "sec_d_6", "second_deriv"),
    Probe("d^2/dx^2 [x^5] =", " 20", " 0", "sec_d_7", "second_deriv"),
    Probe("d^2/dx^2 [x^3] =", " 6", " 0", "sec_d_8", "second_deriv"),
    Probe("f(x) = x^4, f''(x) =", " 12", " 0", "sec_d_9", "second_deriv"),
    Probe("f(x) = x^5, f''(x) =", " 20", " 0", "sec_d_10", "second_deriv"),
    Probe("The second derivative of x^4 + x^3 is", " 12", " 0", "sec_d_11", "second_deriv"),
    Probe("Find the second derivative of x^5:", " 20", " 0", "sec_d_12", "second_deriv"),
    Probe("What is f''(x) if f(x) = x^4?", " 12", " 0", "sec_d_13", "second_deriv"),
    Probe("The second derivative of 3x^3 is", " 18", " 0", "sec_d_14", "second_deriv"),
    Probe("d^2/dx^2 [x^4 + 2x^2] =", " 12", " 0", "sec_d_15", "second_deriv"),
]

INTEGRAL_VARIATIONS = [
    # Original: "The integral of x^2 dx = ", correct " 1", wrong " 2" (1/3 vs 2/3)
    Probe("The integral of x^3 dx =", " 1", " 2", "int_1", "integral"),
    Probe("The integral of x^4 dx =", " 1", " 2", "int_2", "integral"),
    Probe("The integral of x dx =", " 1", " 2", "int_3", "integral"),
    Probe("The integral of x^5 dx =", " 1", " 2", "int_4", "integral"),
    Probe("The integral of 3x^2 dx =", " x", " 2", "int_5", "integral"),
    Probe("The integral of 2x dx =", " x", " 2", "int_6", "integral"),
    Probe("Evaluate the integral of x^2 dx:", " 1", " 2", "int_rephrase_1", "integral"),
    Probe("Find the antiderivative of x^2:", " 1", " 2", "int_rephrase_2", "integral"),
    Probe("What is the integral of x^3 dx?", " 1", " 2", "int_rephrase_3", "integral"),
    Probe("The antiderivative of x^2 is", " 1", " 2", "int_rephrase_4", "integral"),
    Probe("The integral of x^2 with respect to x is", " 1", " 2", "int_rephrase_5", "integral"),
    Probe("Integrate x^3 dx:", " 1", " 2", "int_rephrase_6", "integral"),
    Probe("The indefinite integral of x^4 dx =", " 1", " 2", "int_rephrase_7", "integral"),
    Probe("The integral of x^6 dx =", " 1", " 2", "int_8", "integral"),
    Probe("The integral of 4x^3 dx =", " x", " 2", "int_9", "integral"),
]

PRIME_VARIATIONS = [
    # Original: "Is 97 prime? Answer: ", correct " Yes", wrong " No"
    Probe("Is 101 prime? Answer:", " Yes", " No", "prime_101", "prime"),
    Probe("Is 89 prime? Answer:", " Yes", " No", "prime_89", "prime"),
    Probe("Is 83 prime? Answer:", " Yes", " No", "prime_83", "prime"),
    Probe("Is 71 prime? Answer:", " Yes", " No", "prime_71", "prime"),
    Probe("Is 67 prime? Answer:", " Yes", " No", "prime_67", "prime"),
    Probe("Is 53 prime? Answer:", " Yes", " No", "prime_53", "prime"),
    Probe("Is 107 prime? Answer:", " Yes", " No", "prime_107", "prime"),
    Probe("Is 113 prime? Answer:", " Yes", " No", "prime_113", "prime"),
    # Non-primes (correct answer is No)
    Probe("Is 91 prime? Answer:", " No", " Yes", "notprime_91", "prime"),
    Probe("Is 87 prime? Answer:", " No", " Yes", "notprime_87", "prime"),
    Probe("Is 95 prime? Answer:", " No", " Yes", "notprime_95", "prime"),
    Probe("Is 99 prime? Answer:", " No", " Yes", "notprime_99", "prime"),
    Probe("Is 77 prime? Answer:", " No", " Yes", "notprime_77", "prime"),
    Probe("Is 51 prime? Answer:", " No", " Yes", "notprime_51", "prime"),
    Probe("Is 119 prime? Answer:", " No", " Yes", "notprime_119", "prime"),
]

TRIG_VARIATIONS = [
    # Original: "sin(pi/6) = ", correct " 1", wrong " 0" (1/2 vs 0.523)
    Probe("sin(pi/4) =", " 0", " 1", "trig_sin_pi4", "trig"),  # ~0.707, starts with 0
    Probe("cos(pi/3) =", " 0", " 1", "trig_cos_pi3", "trig"),  # 0.5, starts with 0
    Probe("sin(pi/2) =", " 1", " 0", "trig_sin_pi2", "trig"),  # 1
    Probe("cos(0) =", " 1", " 0", "trig_cos_0", "trig"),  # 1
    Probe("sin(0) =", " 0", " 1", "trig_sin_0", "trig"),  # 0
    Probe("cos(pi) =", " -", " 0", "trig_cos_pi", "trig"),  # -1
    Probe("tan(pi/4) =", " 1", " 0", "trig_tan_pi4", "trig"),  # 1
    Probe("sin(pi) =", " 0", " 1", "trig_sin_pi", "trig"),  # 0
    Probe("cos(pi/2) =", " 0", " 1", "trig_cos_pi2", "trig"),  # 0
    Probe("sin(pi/3) =", " 0", " 1", "trig_sin_pi3", "trig"),  # ~0.866
    Probe("cos(pi/6) =", " 0", " 1", "trig_cos_pi6", "trig"),  # ~0.866
    Probe("tan(0) =", " 0", " 1", "trig_tan_0", "trig"),  # 0
    Probe("sin(2*pi) =", " 0", " 1", "trig_sin_2pi", "trig"),  # 0
    Probe("cos(2*pi) =", " 1", " 0", "trig_cos_2pi", "trig"),  # 1
    Probe("sin(3*pi/2) =", " -", " 0", "trig_sin_3pi2", "trig"),  # -1
]

EXP_DERIV_VARIATIONS = [
    # Original: "d/dx [e^(2x)] = ", correct " 2", wrong " 6"
    Probe("d/dx [e^(3x)] =", " 3", " 9", "exp_d_1", "exp_deriv"),
    Probe("d/dx [e^(4x)] =", " 4", " 16", "exp_d_2", "exp_deriv"),
    Probe("d/dx [e^(5x)] =", " 5", " 25", "exp_d_3", "exp_deriv"),
    Probe("d/dx [e^x] =", " e", " x", "exp_d_4", "exp_deriv"),
    Probe("d/dx [e^(-x)] =", " -", " e", "exp_d_5", "exp_deriv"),
    Probe("The derivative of e^(2x) is", " 2", " 4", "exp_d_rephrase_1", "exp_deriv"),
    Probe("The derivative of e^(3x) is", " 3", " 9", "exp_d_rephrase_2", "exp_deriv"),
    Probe("Differentiate e^(2x):", " 2", " 4", "exp_d_rephrase_3", "exp_deriv"),
    Probe("f(x) = e^(2x), f'(x) =", " 2", " 4", "exp_d_rephrase_4", "exp_deriv"),
    Probe("f(x) = e^(3x), f'(x) =", " 3", " 9", "exp_d_rephrase_5", "exp_deriv"),
    Probe("d/dx [e^(x/2)] =", " 1", " e", "exp_d_6", "exp_deriv"),
    Probe("d/dx [e^(-2x)] =", " -", " 4", "exp_d_7", "exp_deriv"),
    Probe("The derivative of e^x is", " e", " x", "exp_d_8", "exp_deriv"),
    Probe("d/dx [2*e^x] =", " 2", " x", "exp_d_9", "exp_deriv"),
    Probe("d/dx [e^(x^2)] =", " 2", " e", "exp_d_10", "exp_deriv"),
]

ALL_CATEGORIES = [
    ("poly_deriv", POLY_DERIV_VARIATIONS),
    ("second_deriv", SECOND_DERIV_VARIATIONS),
    ("integral", INTEGRAL_VARIATIONS),
    ("prime", PRIME_VARIATIONS),
    ("trig", TRIG_VARIATIONS),
    ("exp_deriv", EXP_DERIV_VARIATIONS),
]


def measure_gaps(backend, probes: list[Probe]) -> dict[str, float]:
    gaps = {}
    for probe in probes:
        tokens = backend.encode(probe.prompt)
        c_id = backend.encode_token(probe.correct_token)
        w_id = backend.encode_token(probe.wrong_token)
        gaps[probe.name] = backend.logit_gap(tokens, c_id, w_id)
    return gaps


def main():
    print("=" * 70)
    print("Bankai Experiment 5: Variation Testing")
    print("Does the calculus patch generalize beyond its 6 training probes?")
    print("=" * 70)

    backend = get_backend("mlx")
    backend.load(MODEL_PATH)
    patch = Patch.load(PATCH_PATH)

    total_improved = 0
    total_unchanged = 0
    total_degraded = 0
    total_probes = 0

    for cat_name, variations in ALL_CATEGORIES:
        print(f"\n{'─'*70}")
        print(f"Category: {cat_name} ({len(variations)} variations)")
        print(f"{'─'*70}")

        # Baseline (no patch)
        baseline = measure_gaps(backend, variations)

        # With patch
        apply_patch(backend, patch)
        patched = measure_gaps(backend, variations)
        remove_patch(backend, patch)

        cat_improved = 0
        cat_degraded = 0
        cat_unchanged = 0

        print(f"  {'Probe':<30} {'Base':>8} {'Patch':>8} {'Delta':>8} {'Result':>10}")
        print(f"  {'-'*66}")

        for probe in variations:
            b = baseline[probe.name]
            p = patched[probe.name]
            delta = p - b

            # Classify: improved if gap increased by >0.1, degraded if decreased by >0.1
            if delta > 0.1:
                result = "IMPROVED"
                cat_improved += 1
            elif delta < -0.1:
                result = "DEGRADED"
                cat_degraded += 1
            else:
                result = "unchanged"
                cat_unchanged += 1

            correct_base = "✓" if b > 0 else "✗"
            correct_patch = "✓" if p > 0 else "✗"

            print(f"  {probe.name:<30} {b:>+7.2f}{correct_base} {p:>+7.2f}{correct_patch} {delta:>+8.2f}  {result}")

        total_improved += cat_improved
        total_degraded += cat_degraded
        total_unchanged += cat_unchanged
        total_probes += len(variations)

        print(f"\n  Summary: {cat_improved} improved, {cat_unchanged} unchanged, {cat_degraded} degraded")

    # Overall
    print(f"\n{'='*70}")
    print(f"OVERALL: {total_probes} variation probes across {len(ALL_CATEGORIES)} categories")
    print(f"  Improved:  {total_improved:>4} ({total_improved/total_probes*100:.1f}%)")
    print(f"  Unchanged: {total_unchanged:>4} ({total_unchanged/total_probes*100:.1f}%)")
    print(f"  Degraded:  {total_degraded:>4} ({total_degraded/total_probes*100:.1f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
