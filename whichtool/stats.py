"""Small statistics helpers: clustered intervals, paired differences, pass^k.

Calls on one job are correlated: repeats of a prompt, and rewordings of the
same job, mostly get the same answer. So every interval treats the job as the
cluster (Miller, "Adding Error Bars to Evals", 2024), and comparisons are
paired by job.
"""

from __future__ import annotations

import math
from collections import defaultdict
from math import comb

from scipy.stats import t as student_t

def t_p_two_sided(t: float, df: int) -> float:
    """Two-sided p-value of Student's t."""
    if df <= 0 or t != t:
        return float("nan")
    return float(2 * student_t.sf(abs(t), df))


def t95(df: int) -> float:
    """Two-sided 95% critical value of Student's t."""
    if df <= 0:
        return float("nan")
    return float(student_t.ppf(0.975, df))


def clustered_mean_ci(values: list[float], clusters: list[str]) -> dict:
    """Mean with a cluster-robust 95% interval (the report clusters by job)."""
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
    """Mean of per-unit differences (a - b) over the units (keys) both sides have.
    The report passes one key per job."""
    common = sorted(set(a) & set(b))
    d = [sum(a[c]) / len(a[c]) - sum(b[c]) / len(b[c]) for c in common if a[c] and b[c]]
    k = len(d)
    if k == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "units": 0, "p": float("nan")}
    mean = sum(d) / k
    if k < 2:
        return {"diff": mean, "lo": float("nan"), "hi": float("nan"), "units": k, "p": float("nan")}
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (k - 1))
    se = sd / math.sqrt(k)
    crit = t95(k - 1)
    p = t_p_two_sided(mean / se, k - 1) if se > 0 else (0.0 if mean != 0 else 1.0)
    return {"diff": mean, "lo": mean - crit * se, "hi": mean + crit * se, "units": k, "p": p}


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
