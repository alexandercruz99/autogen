from __future__ import annotations

import math
import random
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable

from kalshi_bot.ev.combo import ComboLeg, JointResult
from kalshi_bot.money import D, ONE, ZERO, clamp01


@dataclass
class SharedDriverSpec:
    """Reproducible shared-driver simulation for dependent legs.

    Example: game pace driver affecting team win and player prop.
    For weather cities, drivers are independent noise terms unless linked.
    """

    seed: int
    n_samples: int = 5000
    # Each leg: callable(driver_dict) -> settlement in [0,1]
    # Provided by caller as serialized params instead of callables for persistence.


def simulate_independent_binary(
    probs: list[Decimal],
    *,
    seed: int,
    n_samples: int = 5000,
) -> JointResult:
    """Monte Carlo product under independence — only when independence is justified."""
    rng = random.Random(seed)
    ps = [float(clamp01(p)) for p in probs]
    hits = 0
    for _ in range(n_samples):
        ok = True
        for p in ps:
            if rng.random() > p:
                ok = False
                break
        if ok:
            hits += 1
    p_all = D(hits) / D(n_samples)
    # Conservative: Wilson-like downshift using sample SE
    se = math.sqrt(float(p_all) * (1 - float(p_all)) / n_samples) if n_samples else 0.0
    p_cons = clamp01(p_all - D(str(1.96 * se)) - D("0.02"))
    return JointResult(
        p_all=p_all,
        p_all_conservative=p_cons,
        method="mc_independence",
        supported=True,
        details={"n_samples": n_samples, "seed": seed, "se": se},
    )


def simulate_shared_gaussian_driver(
    leg_specs: list[dict[str, Any]],
    *,
    seed: int,
    n_samples: int = 5000,
) -> JointResult:
    """Each leg settles YES if a_i * Z_shared + b_i * Z_i + c_i > 0 (probit-style).

    Pairwise correlation alone is not used as a full multi-leg spec; this is an
    explicit generative model. Fixture/tests supply coefficients.
    """
    rng = random.Random(seed)

    def sample_norm() -> float:
        # Box-Muller
        u1 = max(rng.random(), 1e-12)
        u2 = rng.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)

    settlements: list[float] = []
    for _ in range(n_samples):
        z_shared = sample_norm()
        prod = 1.0
        for spec in leg_specs:
            z_i = sample_norm()
            a = float(spec.get("a_shared", 0.0))
            b = float(spec.get("b_idio", 1.0))
            c = float(spec.get("c", 0.0))
            # Optional scalar settlement mean if DNP-like
            if "forced_settlement" in spec:
                prod *= float(spec["forced_settlement"])
                continue
            yes = (a * z_shared + b * z_i + c) > 0
            prod *= 1.0 if yes else 0.0
        settlements.append(prod)

    mean = D(str(sum(settlements) / n_samples))
    ordered = sorted(settlements)
    idx = max(0, n_samples // 5)
    p_cons = D(str(ordered[idx]))
    return JointResult(
        p_all=mean,
        p_all_conservative=p_cons,
        method="shared_gaussian_driver_mc",
        supported=True,
        details={"n_samples": n_samples, "seed": seed, "legs": len(leg_specs)},
    )


def expected_combo_payout_from_leg_settlements(
    samples: list[list[Decimal]],
) -> Decimal:
    """E[product of leg settlements] including fractional DNP scalars."""
    if not samples:
        return ZERO
    total = ZERO
    for row in samples:
        prod = ONE
        for v in row:
            prod *= D(v)
        total += prod
    return total / D(len(samples))
