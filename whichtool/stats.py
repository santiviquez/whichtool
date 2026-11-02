"""Small statistics helpers: clustered intervals, paired differences, pass^k.

Repeats of one prompt are correlated, so every interval treats the prompt
variant as the cluster (Miller, "Adding Error Bars to Evals", 2024).
"""

from __future__ import annotations

import math
from collections import defaultdict
from math import comb

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes 6.4)."""
    tiny, eps = 1e-300, 3e-14
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1 - x) / b


def t_p_two_sided(t: float, df: int) -> float:
    """Two-sided p-value of Student's t."""
    if df <= 0 or t != t:
        return float("nan")
    return betainc(df / 2, 0.5, df / (df + t * t))


def t95(df: int) -> float:
    """Two-sided 95% critical value of Student's t (bisection on the p-value)."""
    if df <= 0:
        return float("nan")
    lo, hi = 0.0, 1000.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if t_p_two_sided(mid, df) > 0.05:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clustered_mean_ci(values: list[float], clusters: list[str]) -> dict:
    """Mean with a cluster-robust 95% interval (cluster = prompt variant)."""
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "clusters": 0, "se": float("nan")}
    mean = sum(values) / n
    groups: dict[str, list[float]] = defaultdict(list)
    for v, c in zip(values, clusters):
        groups[c].append(v)
    C = len(groups)
    if C < 2:
        se = float("nan")
    else:
        s = sum(sum(v - mean for v in g) ** 2 for g in groups.values())
        se = math.sqrt(C / (C - 1) * s) / n
    crit = t95(C - 1) if C >= 2 else float("nan")
    lo, hi = mean - crit * se, mean + crit * se
    return {"mean": mean, "lo": max(0.0, lo), "hi": min(1.0, hi), "n": n, "clusters": C, "se": se}


def paired_diff_ci(a: dict[str, list[float]], b: dict[str, list[float]]) -> dict:
    """Mean of per-variant differences (a - b) over variants both configs saw."""
    common = sorted(set(a) & set(b))
    d = [sum(a[c]) / len(a[c]) - sum(b[c]) / len(b[c]) for c in common if a[c] and b[c]]
    k = len(d)
    if k == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "variants": 0, "p": float("nan")}
    mean = sum(d) / k
    if k < 2:
        return {"diff": mean, "lo": float("nan"), "hi": float("nan"), "variants": k, "p": float("nan")}
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (k - 1))
    se = sd / math.sqrt(k)
    crit = t95(k - 1)
    p = t_p_two_sided(mean / se, k - 1) if se > 0 else (0.0 if mean != 0 else 1.0)
    return {"diff": mean, "lo": mean - crit * se, "hi": mean + crit * se, "variants": k, "p": p}


def pass_hat_k(c: int, n: int, k: int) -> float:
    """Chance that k independent runs of one prompt are all correct (tau-bench's pass^k)."""
    if k > n:
        return float("nan")
    return comb(c, k) / comb(n, k)


def mean_pass_hat_k(per_variant: dict[str, list[float]], k: int) -> float:
    vals = [
        pass_hat_k(int(sum(v)), len(v), k) for v in per_variant.values() if len(v) >= k
    ]
    return sum(vals) / len(vals) if vals else float("nan")
