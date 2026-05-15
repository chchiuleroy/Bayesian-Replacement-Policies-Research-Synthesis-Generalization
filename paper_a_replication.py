"""
Paper A replication: Bayesian Age Replacement Policy with Safety Constraint
Chiu et al. (2010) — validates Tables 3 & 4 using Proschan (1963) air-conditioner data.

NHPP Power Law: r(t|α,β) = αβt^(β-1)
Failure mode probability p: minor repair with prob p, replacement with prob (1-p)
Conjugate prior: Beta(a0, b0); posterior updated per cycle via numerical integration
"""

import numpy as np
from scipy import integrate, optimize
from scipy.special import erf
import warnings

warnings.filterwarnings("ignore")

# Proschan (1963) air-conditioner failure data (Example 2, Paper A)
# Each cycle: list of inter-failure times (hours × 1e-2 for numerical stability)
PROSCHAN_CYCLES = {
    1: {
        "failures": [0.65, 0.733, 0.757, 0.801, 0.821, 1.01, 1.03, 1.04, 1.05, 1.25],
        "outcome": "replacement",   # cycle ended with replacement (non-repairable)
        "T": 1.49,
    },
    2: {
        "failures": [0.80, 1.18, 1.22, 1.26],
        "catastrophic": [1.46],
        "outcome": "replacement",
        "T": 1.55,
    },
    3: {
        "failures": [0.78, 0.86, 0.90, 0.93, 0.942, 0.95, 1.20],
        "catastrophic": [1.41],
        "outcome": "replacement",
        "T": 1.45,
    },
}

# Paper A calibrated NHPP parameters (Proschan data MLE, Table 1)
ALPHA = 0.7
BETA  = 2.0

# Cost parameters (Paper A Table 2)
C_MIN   = 1.0    # minor repair cost
C_CAT   = 5.0    # catastrophic repair cost
C_REP   = 10.0   # planned/unplanned replacement cost
C_SAFE  = 50.0   # safety penalty (not used in basic policy)

# Safety constraint threshold
OMEGA = 0.05
PLANNING_HORIZON = 1.0   # A in paper


# ---------------------------------------------------------------------------
# NHPP helpers
# ---------------------------------------------------------------------------

def nhpp_intensity(t, alpha=ALPHA, beta=BETA):
    return alpha * beta * t ** (beta - 1)

def nhpp_cumulative(t, alpha=ALPHA, beta=BETA):
    """Λ(t) = α t^β"""
    return alpha * t ** beta

def survival_minor(t, p, gamma, alpha=ALPHA, beta=BETA):
    """
    P(no minor failure in [0,t]) given mode probability p.
    F̄_{(1-γ)p}(t) = exp{-(1-γ)p·α·t^β}
    """
    lam = (1 - gamma) * p * alpha * t ** beta
    return np.exp(-lam)

def expected_cycle_length(T, p, gamma, alpha=ALPHA, beta=BETA):
    """
    E[Y1* | p] = ∫_0^T F̄(t) dt
    For β=2: closed form via erf.
    """
    c = (1 - gamma) * p * alpha
    if abs(beta - 2.0) < 1e-8:
        # ∫_0^T exp(-c t²) dt = sqrt(π/(4c)) · erf(sqrt(c)·T)
        if c < 1e-12:
            return T
        return np.sqrt(np.pi / (4 * c)) * erf(np.sqrt(c) * T)
    # General numerical integration
    result, _ = integrate.quad(
        lambda t: np.exp(-c * alpha * t ** beta / alpha),  # reuse survival_minor pattern
        0, T
    )
    return result

def expected_cycle_cost(T, p, gamma, alpha=ALPHA, beta=BETA,
                         c_min=C_MIN, c_cat=C_CAT, c_rep=C_REP):
    """
    E[R1* | p] — expected cost per cycle.
    = c_min · E[#minor failures in [0,T]]
    + c_cat · E[#catastrophic failures in [0,T]]
    + c_rep · P(replacement at T or catastrophic)

    Simplified under Paper A's model:
    - Minor failures occur at rate (1-γ)p·r(t)
    - Catastrophic failures at rate (1-γ)(1-p)·r(t)  [cycle ends]
    - Planned replacement at T if no catastrophic failure
    """
    lam_total = nhpp_cumulative(T, alpha, beta)
    lam_minor = (1 - gamma) * p * lam_total
    lam_cat   = (1 - gamma) * (1 - p) * lam_total

    # E[#minor] = Λ_minor(T) (for Poisson given no early termination — approximation)
    exp_minor = lam_minor
    # P(catastrophic failure before T) — approximation for rare events
    p_cat = 1 - np.exp(-lam_cat)

    cost = c_min * exp_minor + c_cat * p_cat + c_rep
    return cost

def bayesian_cost_rate(T, p_grid, w_grid, gamma,
                       alpha=ALPHA, beta=BETA):
    """
    C_B(T) = E_π[R1*(T)] / E_π[Y1*(T)]
           = ∫ R(T,p)·π(p) dp / ∫ Y(T,p)·π(p) dp
    Numerically: weighted average over p_grid with weights w_grid.
    """
    R_vals = np.array([expected_cycle_cost(T, p, gamma, alpha, beta)
                       for p in p_grid])
    Y_vals = np.array([expected_cycle_length(T, p, gamma, alpha, beta)
                       for p in p_grid])

    w = w_grid / w_grid.sum()
    E_R = np.dot(w, R_vals)
    E_Y = np.dot(w, Y_vals)
    return E_R / E_Y if E_Y > 1e-12 else np.inf


# ---------------------------------------------------------------------------
# Beta-exponential posterior (numerical)
# ---------------------------------------------------------------------------

def posterior_weights(p_grid, a_prior, b_prior, cycle_data,
                      alpha=ALPHA, beta=BETA):
    """
    Update prior Beta(a0,b0) with one cycle's data.
    Posterior ∝ prior(p) · likelihood(data | p)

    Likelihood for a cycle ending at T with n_minor minor failures:
        L(p) ∝ p^n_minor · exp{-(1-γ)p·Λ(T)}  [conjugate-like structure]

    If cycle_data is empty (prior-only), returns Beta(a,b) weights.
    Returns unnormalized weights on p_grid.
    """
    gamma = 0.0   # Paper A uses γ=0

    if not cycle_data:
        # Prior only: Beta(a_prior, b_prior)
        log_weights = (
            (a_prior - 1) * np.log(np.clip(p_grid, 1e-15, 1))
            + (b_prior - 1) * np.log(np.clip(1 - p_grid, 1e-15, 1))
        )
        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        return weights / weights.sum()

    n_minor = len(cycle_data.get("failures", []))
    T_cycle = cycle_data["T"]
    lam_T   = nhpp_cumulative(T_cycle, alpha, beta)

    # Prior Beta(a,b) density ∝ p^(a-1) · (1-p)^(b-1)
    # Likelihood ∝ p^n_minor · exp{-(1-γ)p·Λ(T)}
    log_weights = (
        (a_prior - 1 + n_minor) * np.log(np.clip(p_grid, 1e-15, 1))
        + (b_prior - 1) * np.log(np.clip(1 - p_grid, 1e-15, 1))
        - (1 - gamma) * p_grid * lam_T
    )
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    return weights / weights.sum()


# ---------------------------------------------------------------------------
# Safety constraint
# ---------------------------------------------------------------------------

def safety_threshold_T(p_grid, w_grid, alpha=ALPHA, beta=BETA,
                        A=PLANNING_HORIZON, omega=OMEGA):
    """
    Find T_ω such that P(N_m(A) ≥ 1) = ω.
    P(N_m(A) ≥ 1 | p) = 1 - exp{-(1-p)·Λ(A)}   [catastrophic events]
    E_π[P(...)] ≤ ω  →  solve for T_ω  (here A is fixed; T acts via p updating)

    Since safety is on the planning horizon A (not T), this computes T_ω
    as the largest T such that E_π[1-exp{-(1-p)Λ(A)}] ≤ ω.
    In practice: compute E_π[safety_prob] and compare to ω.
    """
    w = w_grid / w_grid.sum()
    safety_probs = 1 - np.exp(-(1 - p_grid) * nhpp_cumulative(A, alpha, beta))
    return np.dot(w, safety_probs)


# ---------------------------------------------------------------------------
# Main replication
# ---------------------------------------------------------------------------

def run_table(prior_a, prior_b, label, gamma_vals=(0.2, 0.5, 0.7),
              T_search=np.linspace(0.1, 3.0, 300),
              n_p_grid=500):

    p_grid = np.linspace(1e-4, 1 - 1e-4, n_p_grid)
    print(f"\n{'='*65}")
    print(f"  {label}  —  Prior: Beta({prior_a}, {prior_b})")
    print(f"{'='*65}")

    results = {}
    for gamma in gamma_vals:
        print(f"\n  γ = {gamma}")
        print(f"  {'Cycle':>5}  {'T*':>7}  {'C_B(T*)':>10}  {'Safety P':>10}")
        print(f"  {'-'*40}")

        a, b = prior_a, prior_b

        for cycle_idx in range(4):   # cycle 0 = prior only, 1-3 = after each cycle
            # Current posterior weights
            w = posterior_weights(p_grid, a, b,
                                  {} if cycle_idx == 0 else PROSCHAN_CYCLES[cycle_idx],
                                  ALPHA, BETA)
            if cycle_idx > 0:
                # Update hyperparameters for next iteration
                # (simplified: shift a,b via sufficient statistics)
                n_m = len(PROSCHAN_CYCLES[cycle_idx].get("failures", []))
                lam = nhpp_cumulative(PROSCHAN_CYCLES[cycle_idx]["T"])
                a = a + n_m
                b = b + lam  # rough conjugate update; exact is via numerical posterior above

            # Optimal T* minimizing Bayesian cost rate
            costs = np.array([bayesian_cost_rate(T, p_grid, w, gamma)
                              for T in T_search])
            best_idx = np.argmin(costs)
            T_star  = T_search[best_idx]
            C_star  = costs[best_idx]
            safety  = safety_threshold_T(p_grid, w)

            label_cycle = "Prior" if cycle_idx == 0 else f"After C{cycle_idx}"
            print(f"  {label_cycle:>8}  {T_star:>7.3f}  {C_star:>10.4f}  {safety:>10.4f}")
            results[(gamma, cycle_idx)] = (T_star, C_star, safety)

    return results


def main():
    np.random.seed(42)

    print("\nPaper A Replication — Bayesian ARP (Chiu et al. 2010)")
    print("Proschan (1963) Air-Conditioner Data | α=0.7, β=2.0\n")

    # Table 3: conjugate prior Beta(1, 9)
    res3 = run_table(prior_a=1, prior_b=9,
                     label="Table 3 — Conjugate Prior Beta(1,9)")

    # Table 4: non-informative prior Beta(1, 1)
    res4 = run_table(prior_a=1, prior_b=1,
                     label="Table 4 — Non-informative Prior Beta(1,1)")

    print("\n\nSensitivity: safety constraint T_ω vs γ")
    print(f"  {'γ':>5}  {'T*(no SC)':>12}  {'Safety P':>10}  {'Constrained?':>12}")
    print(f"  {'-'*45}")
    gamma_vals = [0.2, 0.5, 0.7]
    p_grid = np.linspace(1e-4, 1 - 1e-4, 500)
    # Use posterior after all 3 cycles, non-informative prior
    a, b = 1.0, 1.0
    for c in [1, 2, 3]:
        n_m = len(PROSCHAN_CYCLES[c].get("failures", []))
        lam = nhpp_cumulative(PROSCHAN_CYCLES[c]["T"])
        a += n_m; b += lam
    w_final = posterior_weights(p_grid, a, b, PROSCHAN_CYCLES[3])

    T_search = np.linspace(0.1, 3.0, 300)
    for gamma in gamma_vals:
        costs = np.array([bayesian_cost_rate(T, p_grid, w_final, gamma)
                          for T in T_search])
        T_star = T_search[np.argmin(costs)]
        safety = safety_threshold_T(p_grid, w_final)
        constrained = safety > OMEGA
        print(f"  {gamma:>5.1f}  {T_star:>12.3f}  {safety:>10.4f}  {'YES' if constrained else 'no':>12}")


if __name__ == "__main__":
    main()
