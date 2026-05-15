"""
Parallel-Series System Simulation for GMBRM
============================================
System topology:
  k subsystems in SERIES  (any subsystem failure → system failure)
  Each subsystem: r components in PARALLEL (subsystem fails when ALL r fail)

Special case: k=1, r=1, a=b=1 → single-unit model (Paper A replication check)

NHPP Power Law component failure intensity: r(t) = αβt^(β-1)
Geometric process deterioration (Paper C):
  n-th working time: F_n(t) = F(a^(n-1) t)   [a>1 → deterioration]
  n-th repair time:  G_n(t) = G(b^(n-1) t)   [b<1 → faster repair]
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

# Default parameters (Proschan calibrated)
ALPHA  = 0.7
BETA   = 2.0
C_MIN  = 1.0     # minor repair cost per event
C_CAT  = 5.0     # catastrophic (repairable) cost
C_REP  = 10.0    # replacement cost (planned or unplanned)
OMEGA  = 0.05    # safety constraint threshold
A_HOR  = 1.0     # planning horizon A


# ---------------------------------------------------------------------------
# NHPP helpers
# ---------------------------------------------------------------------------

def nhpp_next_event(t_current: float, scale: float,
                    alpha: float, beta: float, rng: np.random.Generator) -> float:
    """
    Sample next NHPP event time after t_current via integrated hazard inversion.
    Λ(t) = α·t^β  →  Λ^{-1}(u) = (u/α)^{1/β}
    Gap: solve Λ(t_next) - Λ(t_current) = Exp(1)
    """
    gap = rng.exponential(1.0)
    lam_now = alpha * scale * t_current ** beta
    lam_next = lam_now + gap
    t_next = (lam_next / (alpha * scale)) ** (1.0 / beta)
    return t_next


def repair_duration(b_factor: float, rng: np.random.Generator,
                    base_mean: float = 0.05) -> float:
    """Exponential repair time, mean = base_mean / b_factor (geometric process)."""
    return rng.exponential(base_mean / b_factor)


# ---------------------------------------------------------------------------
# Component state
# ---------------------------------------------------------------------------

@dataclass
class Component:
    subsys_id: int
    comp_id:   int
    n_repairs: int = 0        # repair count (drives deterioration)
    t_failed:  Optional[float] = None
    alive:     bool = True

    def deterioration_scale(self, a: float) -> float:
        """Scale factor for n-th inter-failure time: a^(n-1)."""
        return a ** self.n_repairs

    def repair_scale(self, b: float) -> float:
        return b ** self.n_repairs


@dataclass
class Subsystem:
    sub_id: int
    components: List[Component]

    @property
    def alive(self) -> bool:
        return any(c.alive for c in self.components)


# ---------------------------------------------------------------------------
# Failure mode classifier (maps network event → GMBRM mode)
# ---------------------------------------------------------------------------

def classify_failure_mode(p: float, rng: np.random.Generator) -> str:
    """
    Given failure mode probability p (minor repair fraction):
    - Draw Bernoulli(p) → "minor" or "catastrophic"
    "catastrophic" here means repairable-catastrophic (Paper A/B mode).
    Non-repairable replacement is decided at the system level.
    """
    return "minor" if rng.random() < p else "catastrophic"


# ---------------------------------------------------------------------------
# One simulation cycle
# ---------------------------------------------------------------------------

def simulate_cycle(
    T: float,
    k: int, r: int,
    p: float,
    alpha: float, beta: float,
    a: float, b: float,
    c_min: float, c_cat: float, c_rep: float,
    rng: np.random.Generator,
) -> Tuple[float, float, dict]:
    """
    Simulate one replacement cycle for the parallel-series system.

    Returns:
        cycle_length : float  — time until replacement
        cycle_cost   : float  — total cost incurred
        stats        : dict   — breakdown for analysis
    """
    # Initialise components
    subsystems = [
        Subsystem(
            sub_id=sid,
            components=[Component(subsys_id=sid, comp_id=cid)
                        for cid in range(r)]
        )
        for sid in range(k)
    ]

    t = 0.0
    cost = 0.0
    stats = {"minor": 0, "catastrophic": 0, "replacements": 0}

    # Priority queue: (next_failure_time, subsys_id, comp_id)
    import heapq
    events = []
    for sub in subsystems:
        for comp in sub.components:
            t_fail = nhpp_next_event(1e-6, comp.deterioration_scale(a),
                                     alpha, beta, rng)
            heapq.heappush(events, (t_fail, sub.sub_id, comp.comp_id))

    while events:
        t_event, sid, cid = heapq.heappop(events)

        if t_event >= T:
            # Planned replacement at T
            cost += c_rep
            stats["replacements"] += 1
            return T, cost, stats

        comp = subsystems[sid].components[cid]
        if not comp.alive:
            continue   # stale event after component already dead

        t = t_event
        mode = classify_failure_mode(p, rng)

        if mode == "minor":
            # Minor repair: component restored, time counter reset
            cost += c_min
            stats["minor"] += 1
            comp.n_repairs += 1
            # Schedule next failure
            t_next = nhpp_next_event(t, comp.deterioration_scale(a),
                                     alpha, beta, rng)
            heapq.heappush(events, (t_next, sid, cid))

        else:
            # Catastrophic failure: component dies
            comp.alive = False
            comp.t_failed = t
            cost += c_cat
            stats["catastrophic"] += 1

            # Check if subsystem is now dead
            if not subsystems[sid].alive:
                # Series connection: system fails → unplanned replacement
                cost += c_rep
                stats["replacements"] += 1
                return t, cost, stats

            # Subsystem still alive (other parallel components running)
            # No new event for dead component

    # Should not reach here normally
    return T, cost, stats


# ---------------------------------------------------------------------------
# Monte Carlo cost rate estimator
# ---------------------------------------------------------------------------

def estimate_cost_rate(
    T: float,
    k: int = 1, r: int = 1,
    p: float = 0.7,
    alpha: float = ALPHA, beta: float = BETA,
    a: float = 1.0, b: float = 1.0,
    c_min: float = C_MIN, c_cat: float = C_CAT, c_rep: float = C_REP,
    n_cycles: int = 5000,
    seed: Optional[int] = 42,
) -> Tuple[float, float, float]:
    """
    Returns (mean_cost_rate, std_cost_rate, mean_cycle_length).
    Cost rate by renewal reward theorem: E[cost] / E[length].
    """
    rng = np.random.default_rng(seed)
    lengths, costs = [], []

    for _ in range(n_cycles):
        length, cost, _ = simulate_cycle(
            T, k, r, p, alpha, beta, a, b, c_min, c_cat, c_rep, rng
        )
        lengths.append(length)
        costs.append(cost)

    lengths = np.array(lengths)
    costs   = np.array(costs)

    mean_rate = costs.mean() / lengths.mean()
    # Delta method variance
    E_C, E_L = costs.mean(), lengths.mean()
    var_C = costs.var() / n_cycles
    var_L = lengths.var() / n_cycles
    cov_CL = np.cov(costs, lengths)[0, 1] / n_cycles
    var_rate = (var_C / E_L**2
                + E_C**2 * var_L / E_L**4
                - 2 * E_C * cov_CL / E_L**3)
    std_rate = np.sqrt(max(var_rate, 0))

    return mean_rate, std_rate, lengths.mean()


# ---------------------------------------------------------------------------
# Optimal T* search
# ---------------------------------------------------------------------------

def optimal_T(k, r, p, alpha=ALPHA, beta=BETA,
              a=1.0, b=1.0, T_grid=None, n_cycles=3000, seed=42):
    if T_grid is None:
        T_grid = np.linspace(0.3, 3.0, 30)
    rates = []
    for T in T_grid:
        rate, _, _ = estimate_cost_rate(T, k, r, p, alpha, beta, a, b,
                                        n_cycles=n_cycles, seed=seed)
        rates.append(rate)
    rates = np.array(rates)
    best = np.argmin(rates)
    return T_grid[best], rates[best], rates


# ---------------------------------------------------------------------------
# Safety constraint check
# ---------------------------------------------------------------------------

def safety_probability(T_star: float, k: int, r: int, p: float,
                        alpha: float = ALPHA, beta: float = BETA,
                        A: float = A_HOR, n_cycles: int = 2000,
                        seed: int = 0) -> float:
    """
    Estimate P(system failure in [0,A]) via Monte Carlo.
    Compare against ω; if P > ω, apply constraint T_SC = min(T*, T_ω).
    """
    rng = np.random.default_rng(seed)
    failures = 0
    for _ in range(n_cycles):
        length, _, _ = simulate_cycle(T_star, k, r, p, alpha, beta,
                                       1.0, 1.0, C_MIN, C_CAT, C_REP, rng)
        if length < A:   # system failed before horizon
            failures += 1
    return failures / n_cycles


# ---------------------------------------------------------------------------
# Sensitivity experiments
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    # label,                k, r,  p,    a,    b,   description
    ("Baseline (k=1,r=1)",  1, 1, 0.7, 1.00, 1.00, "Single unit, Paper A check"),
    ("Series k=3",          3, 1, 0.7, 1.00, 1.00, "Pure series; k↑ → system fragile"),
    ("Parallel r=3",        1, 3, 0.7, 1.00, 1.00, "Pure parallel; r↑ → redundancy"),
    ("2×2 network",         2, 2, 0.7, 1.00, 1.00, "Balanced parallel-series"),
    ("Deterioration a=1.1", 2, 2, 0.7, 1.10, 0.95, "Geometric process aging"),
    ("Co-policy (low p)",   2, 2, 0.3, 1.05, 0.98, "Low minor-repair fraction"),
]


def run_experiments(T_grid=None, n_cycles=3000):
    if T_grid is None:
        T_grid = np.linspace(0.3, 3.0, 25)

    print("\n" + "="*78)
    print("  Parallel-Series GMBRM Simulation  |  Proschan α=0.7, β=2.0")
    print("="*78)
    print(f"  {'Experiment':<25}  {'k':>3} {'r':>3} {'p':>5}  {'T*':>6}  "
          f"{'C_B(T*)':>9}  {'Safety P':>9}  {'SC?':>4}")
    print(f"  {'-'*72}")

    for label, k, r, p, a, b, _ in EXPERIMENTS:
        T_star, C_star, _ = optimal_T(k, r, p, a=a, b=b,
                                       T_grid=T_grid, n_cycles=n_cycles)
        safety = safety_probability(T_star, k, r, p, n_cycles=min(n_cycles, 1500))
        sc = "YES" if safety > OMEGA else "no"
        print(f"  {label:<25}  {k:>3} {r:>3} {p:>5.2f}  {T_star:>6.2f}  "
              f"{C_star:>9.4f}  {safety:>9.4f}  {sc:>4}")


def run_redundancy_sweep():
    """Sweep r (parallel redundancy) and show T*, cost rate, safety."""
    print("\n--- Redundancy Sweep (k=2, p=0.7, no deterioration) ---")
    print(f"  {'r':>4}  {'T*':>8}  {'C_B(T*)':>10}  {'Safety P':>10}")
    print(f"  {'-'*38}")
    T_grid = np.linspace(0.3, 3.0, 25)
    for r in range(1, 6):
        T_star, C_star, _ = optimal_T(k=2, r=r, p=0.7, T_grid=T_grid, n_cycles=2000)
        safety = safety_probability(T_star, k=2, r=r, p=0.7, n_cycles=1000)
        print(f"  {r:>4}  {T_star:>8.3f}  {C_star:>10.4f}  {safety:>10.4f}")


def run_deterioration_sweep():
    """Sweep deterioration parameter a and show optimal policy shift."""
    print("\n--- Deterioration Sweep (k=2, r=2, p=0.7, b=1/a) ---")
    print(f"  {'a':>6}  {'T*':>8}  {'C_B(T*)':>10}  {'Safety P':>10}")
    print(f"  {'-'*40}")
    T_grid = np.linspace(0.3, 3.0, 25)
    for a in [1.0, 1.05, 1.10, 1.20, 1.30]:
        b = 1.0 / a
        T_star, C_star, _ = optimal_T(k=2, r=2, p=0.7, a=a, b=b,
                                       T_grid=T_grid, n_cycles=2000)
        safety = safety_probability(T_star, k=2, r=2, p=0.7, n_cycles=1000)
        print(f"  {a:>6.2f}  {T_star:>8.3f}  {C_star:>10.4f}  {safety:>10.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel-Series GMBRM Simulation")
    parser.add_argument("--mode", choices=["experiments", "redundancy", "deterioration", "all"],
                        default="all")
    parser.add_argument("--cycles", type=int, default=3000,
                        help="Monte Carlo cycles per T evaluation")
    args = parser.parse_args()

    T_grid = np.linspace(0.3, 3.0, 25)

    if args.mode in ("experiments", "all"):
        run_experiments(T_grid=T_grid, n_cycles=args.cycles)

    if args.mode in ("redundancy", "all"):
        run_redundancy_sweep()

    if args.mode in ("deterioration", "all"):
        run_deterioration_sweep()
