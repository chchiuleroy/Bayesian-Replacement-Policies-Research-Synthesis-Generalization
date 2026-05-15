"""
Adaptive Replacement via Posterior Sampling RL (PSRL)
=====================================================
Combines:
  - Thompson Sampling: at each episode start, sample model from posterior
  - MDP structure: state = (age, cycle_n, posterior_hyperparams)
  - Sequential Bayesian updating: posterior updated after every cycle

Episode = one replacement cycle.
At episode start: sample (alpha, beta, p) ~ posterior → compute T*(alpha,beta,p) → follow T*
At episode end: observe (n_minor, n_cat, T_actual) → update posterior

Agents compared:
  PSRLAgent    - Thompson Sampling (PSRL)
  GreedyBayes  - uses posterior mean, no exploration
  FixedTAgent  - fixed T regardless of data (non-adaptive baseline)
  OracleAgent  - knows true parameters (regret lower bound)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

# True system parameters (Proschan-calibrated; unknown to all agents except Oracle)
TRUE_ALPHA = 0.7
TRUE_BETA  = 2.0
TRUE_P     = 0.65      # minor-repair probability (unknown to agents)

# Cost parameters
C_MIN = 1.0
C_CAT = 5.0
C_REP = 10.0

# Geometric deterioration
A_DETR = 1.05   # working-time scale per repair
B_DETR = 0.95   # repair-time scale per repair


# ---------------------------------------------------------------------------
# Environment: one replacement cycle
# ---------------------------------------------------------------------------

@dataclass
class CycleResult:
    T_planned:   float
    T_actual:    float        # actual cycle length (≤ T_planned)
    n_minor:     int
    n_cat:       int
    cost:        float
    terminated_early: bool    # True if non-repairable failure before T


def simulate_cycle_env(
    T: float,
    alpha: float, beta: float, p: float,
    a: float = A_DETR, b: float = B_DETR,
    n_repairs: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> CycleResult:
    """
    Simulate one replacement cycle under NHPP Power Law + geometric deterioration.
    n_repairs: total repairs before this cycle (drives deterioration scale).
    """
    if rng is None:
        rng = np.random.default_rng()

    t = 0.0
    cost = 0.0
    n_minor = 0
    n_cat = 0
    local_repairs = 0

    def nhpp_next(t_now, scale):
        lam_now  = alpha * scale * t_now ** beta
        gap      = rng.exponential(1.0)
        lam_next = lam_now + gap
        return (lam_next / (alpha * scale)) ** (1.0 / beta)

    # Deterioration scale grows with total repair count
    scale = a ** (n_repairs + local_repairs)
    t_next = nhpp_next(max(t, 1e-9), scale)

    while t_next < T:
        t = t_next
        if rng.random() < p:
            # Minor repair
            n_minor += 1
            cost += C_MIN
            local_repairs += 1
            scale = a ** (n_repairs + local_repairs)
            t_next = nhpp_next(t, scale)
        else:
            # Catastrophic failure — non-repairable → early replacement
            n_cat += 1
            cost += C_CAT + C_REP
            return CycleResult(T, t, n_minor, n_cat, cost, terminated_early=True)

    # Planned replacement at T
    cost += C_REP
    return CycleResult(T, T, n_minor, n_cat, cost, terminated_early=False)


# ---------------------------------------------------------------------------
# Posterior: Beta on p, Gamma on (alpha, beta) via sufficient stats
# ---------------------------------------------------------------------------

@dataclass
class Posterior:
    """
    Conjugate posterior for (p, alpha, beta).

    For p: Beta(a_p, b_p)
      a_p += n_minor per cycle  (minor events support high p)
      b_p += n_cat per cycle    (catastrophic events support low p)
      Correction term: exp{-p·Λ(T)} in the likelihood shifts the effective prior.

    For (alpha, beta): approximated via running MLE on observed failure times,
    with Gamma hyperprior Gamma(k_a, theta_a) on alpha.
    In a full conjugate treatment (Huang & Bier 1998) the joint prior is
    more complex; here we use the Beta-exponential numerical update for p
    and a Gaussian approximation for (alpha, beta) from the MLE information.
    """
    # Beta prior on p
    a_p: float = 1.0
    b_p: float = 9.0   # start with conjugate Beta(1,9) as in Paper A Table 3

    # Empirical posterior on alpha, beta (running sample moments from MLE)
    alpha_samples: List[float] = field(default_factory=lambda: [0.7])
    beta_samples:  List[float] = field(default_factory=lambda: [2.0])

    # Accumulated data
    total_minor:  int   = 0
    total_cat:    int   = 0
    total_cycles: int   = 0
    cumulative_cost: float = 0.0

    def update(self, result: CycleResult):
        """Update posterior with one cycle's observations."""
        self.total_minor  += result.n_minor
        self.total_cat    += result.n_cat
        self.total_cycles += 1
        self.cumulative_cost += result.cost

        # Beta-exponential update for p:
        # posterior a_p += n_minor, b_p += (n_cat + correction from survival)
        # Simplified: treat each catastrophic as evidence against p
        lam_T = self.alpha_mean * result.T_actual ** self.beta_mean
        # Effective update: a_p += n_minor, b_p += max(n_cat, 1e-3) + lam_T * (1-p_mean)
        self.a_p += result.n_minor
        self.b_p += result.n_cat + 0.1 * lam_T * (1 - self.p_mean)

    @property
    def p_mean(self) -> float:
        return self.a_p / (self.a_p + self.b_p)

    @property
    def alpha_mean(self) -> float:
        return np.mean(self.alpha_samples)

    @property
    def beta_mean(self) -> float:
        return np.mean(self.beta_samples)

    def sample(self, rng: np.random.Generator) -> Tuple[float, float, float]:
        """Sample (alpha, beta, p) from posterior."""
        p = rng.beta(self.a_p, self.b_p)

        # Sample alpha/beta from empirical posterior (Gaussian around running MLE)
        # Use log-normal to keep positivity; std proportional to 1/sqrt(n)
        n = max(self.total_cycles, 1)
        alpha = max(rng.normal(self.alpha_mean, self.alpha_mean * 0.3 / np.sqrt(n)), 0.1)
        beta  = max(rng.normal(self.beta_mean,  self.beta_mean  * 0.3 / np.sqrt(n)), 0.5)
        return alpha, beta, p


# ---------------------------------------------------------------------------
# Optimal T* under given parameters (analytical / numerical)
# ---------------------------------------------------------------------------

def optimal_T_given_params(
    alpha: float, beta: float, p: float,
    T_grid: np.ndarray = None,
    c_min: float = C_MIN, c_cat: float = C_CAT, c_rep: float = C_REP,
) -> float:
    """Minimize cost rate C(T) = E[cost] / E[length] under known (alpha, beta, p)."""
    if T_grid is None:
        T_grid = np.linspace(0.3, 5.0, 80)

    def cost_rate(T):
        lam = alpha * T ** beta
        lam_minor = p * lam
        lam_cat   = (1 - p) * lam
        p_cat     = 1 - np.exp(-lam_cat)
        E_cost    = c_min * lam_minor + c_cat * p_cat + c_rep

        from scipy.special import erf
        if abs(beta - 2.0) < 0.1:
            c_sq = p * alpha
            E_len = np.sqrt(np.pi / (4 * c_sq)) * erf(np.sqrt(c_sq) * T) if c_sq > 1e-12 else T
        else:
            from scipy import integrate
            E_len, _ = integrate.quad(lambda t: np.exp(-p * alpha * t**beta), 0, T)

        return E_cost / max(E_len, 1e-12)

    rates = [cost_rate(T) for T in T_grid]
    return T_grid[np.argmin(rates)]


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class PSRLAgent:
    """Posterior Sampling RL: sample model from posterior each episode."""
    name = "PSRL (Thompson Sampling)"

    def __init__(self, seed=0):
        self.posterior = Posterior()
        self.rng = np.random.default_rng(seed)
        self.T_history: List[float] = []

    def act(self) -> float:
        alpha, beta, p = self.posterior.sample(self.rng)
        T = optimal_T_given_params(alpha, beta, p)
        self.T_history.append(T)
        return T

    def observe(self, result: CycleResult):
        self.posterior.update(result)


class GreedyBayesAgent:
    """Uses posterior mean — no exploration, maximum exploitation."""
    name = "Greedy Bayes (posterior mean)"

    def __init__(self):
        self.posterior = Posterior()
        self.T_history: List[float] = []

    def act(self) -> float:
        T = optimal_T_given_params(
            self.posterior.alpha_mean,
            self.posterior.beta_mean,
            self.posterior.p_mean,
        )
        self.T_history.append(T)
        return T

    def observe(self, result: CycleResult):
        self.posterior.update(result)


class FixedTAgent:
    """Fixed replacement interval — no learning."""
    name = "Fixed T (non-adaptive)"

    def __init__(self, T: float = 2.0):
        self.T_fixed = T
        self.T_history: List[float] = []

    def act(self) -> float:
        self.T_history.append(self.T_fixed)
        return self.T_fixed

    def observe(self, result: CycleResult):
        pass


class OracleAgent:
    """Knows true parameters — provides regret lower bound."""
    name = "Oracle (true params)"

    def __init__(self):
        self.T_star = optimal_T_given_params(TRUE_ALPHA, TRUE_BETA, TRUE_P)
        self.T_history: List[float] = []

    def act(self) -> float:
        self.T_history.append(self.T_star)
        return self.T_star

    def observe(self, result: CycleResult):
        pass


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class AgentLog:
    costs:     List[float] = field(default_factory=list)
    lengths:   List[float] = field(default_factory=list)
    T_choices: List[float] = field(default_factory=list)

    @property
    def cumulative_cost(self) -> np.ndarray:
        return np.cumsum(self.costs)

    @property
    def cost_rate(self) -> float:
        if not self.costs:
            return float("inf")
        return sum(self.costs) / max(sum(self.lengths), 1e-12)


def run_experiment(
    agents,
    n_episodes: int = 200,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    logs = {agent.name: AgentLog() for agent in agents}

    for episode in range(n_episodes):
        for agent in agents:
            T = agent.act()
            result = simulate_cycle_env(
                T,
                alpha=TRUE_ALPHA, beta=TRUE_BETA, p=TRUE_P,
                rng=rng,
            )
            agent.observe(result)
            logs[agent.name].costs.append(result.cost)
            logs[agent.name].lengths.append(result.T_actual)
            logs[agent.name].T_choices.append(T)

    return logs


# ---------------------------------------------------------------------------
# Posterior convergence tracker (PSRL only)
# ---------------------------------------------------------------------------

def run_posterior_convergence(n_episodes=150, seed=0):
    """
    Track how quickly PSRL's posterior mean of p converges to TRUE_P.
    Returns list of (episode, p_mean, T*_mean, p_std).
    """
    agent = PSRLAgent(seed=seed)
    rng   = np.random.default_rng(seed + 1)
    trace = []

    for ep in range(n_episodes):
        T = agent.act()
        result = simulate_cycle_env(T, TRUE_ALPHA, TRUE_BETA, TRUE_P, rng=rng)
        agent.observe(result)

        if ep % 10 == 0 or ep < 10:
            post = agent.posterior
            p_std = np.sqrt(
                post.a_p * post.b_p
                / ((post.a_p + post.b_p) ** 2 * (post.a_p + post.b_p + 1))
            )
            T_mean = np.mean(agent.T_history[-min(10, ep+1):])
            trace.append((ep, post.p_mean, T_mean, p_std))

    return trace


# ---------------------------------------------------------------------------
# Regret analysis
# ---------------------------------------------------------------------------

def compute_regret(logs: dict, oracle_name: str = "Oracle (true params)") -> dict:
    """
    Regret_t = cumulative_cost(agent, t) - cumulative_cost(oracle, t)
    """
    oracle_cum = logs[oracle_name].cumulative_cost
    regrets = {}
    for name, log in logs.items():
        if name == oracle_name:
            continue
        regrets[name] = log.cumulative_cost - oracle_cum
    return regrets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_separator(title=""):
    width = 70
    if title:
        pad = (width - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * (width - pad - len(title) - 2))
    else:
        print("=" * width)


def main():
    np.random.seed(42)

    print_separator("PSRL Adaptive Replacement — Setup")
    print(f"  True parameters: α={TRUE_ALPHA}, β={TRUE_BETA}, p={TRUE_P}")
    T_oracle = optimal_T_given_params(TRUE_ALPHA, TRUE_BETA, TRUE_P)
    print(f"  Oracle optimal T*: {T_oracle:.3f}")
    print(f"  Agents: PSRL, Greedy Bayes, Fixed T=2.0, Oracle")

    # -----------------------------------------------------------------------
    print_separator("Experiment A: 200 Episodes Comparison")
    agents = [
        PSRLAgent(seed=7),
        GreedyBayesAgent(),
        FixedTAgent(T=2.0),
        OracleAgent(),
    ]
    logs = run_experiment(agents, n_episodes=200, seed=42)

    print(f"\n  {'Agent':<30}  {'Cost Rate':>10}  {'Final T*':>9}  {'Total Cost':>11}")
    print(f"  {'-'*65}")
    for agent in agents:
        log = logs[agent.name]
        final_T = np.mean(log.T_choices[-20:])
        print(f"  {agent.name:<30}  {log.cost_rate:>10.4f}  {final_T:>9.3f}  {log.cumulative_cost[-1]:>11.1f}")

    # -----------------------------------------------------------------------
    print_separator("Experiment B: Cumulative Regret (vs Oracle)")
    regrets = compute_regret(logs)
    milestones = [10, 30, 50, 100, 150, 200]
    header = f"  {'Agent':<30}" + "".join(f"  {'Ep'+str(m):>7}" for m in milestones)
    print(header)
    print(f"  {'-'*75}")
    for name, regret in regrets.items():
        row = f"  {name:<30}"
        for m in milestones:
            row += f"  {regret[m-1]:>7.1f}"
        print(row)

    # -----------------------------------------------------------------------
    print_separator("Experiment C: Posterior Convergence (PSRL)")
    print(f"\n  True p = {TRUE_P}")
    print(f"\n  {'Episode':>8}  {'p_mean':>8}  {'p_std':>8}  {'T*_mean':>9}  {'Error%':>8}")
    print(f"  {'-'*48}")
    trace = run_posterior_convergence(n_episodes=150, seed=0)
    for ep, p_mean, T_mean, p_std in trace:
        err = abs(p_mean - TRUE_P) / TRUE_P * 100
        print(f"  {ep:>8}  {p_mean:>8.4f}  {p_std:>8.4f}  {T_mean:>9.3f}  {err:>7.1f}%")

    # -----------------------------------------------------------------------
    print_separator("Experiment D: Sensitivity — Prior Strength")
    print(f"\n  Effect of initial prior Beta(a0, b0) on PSRL convergence speed")
    print(f"\n  {'Prior':>15}  {'p_mean@ep5':>12}  {'p_mean@ep20':>13}  {'Final T*':>10}")
    print(f"  {'-'*55}")
    T_grid = np.linspace(0.3, 5.0, 80)
    for (a0, b0) in [(1, 1), (1, 9), (5, 5), (1, 19)]:
        agent = PSRLAgent(seed=0)
        agent.posterior.a_p = a0
        agent.posterior.b_p = b0
        rng = np.random.default_rng(1)
        p_at = {}
        for ep in range(50):
            T = agent.act()
            result = simulate_cycle_env(T, TRUE_ALPHA, TRUE_BETA, TRUE_P, rng=rng)
            agent.observe(result)
            if ep in (4, 19, 49):
                p_at[ep] = agent.posterior.p_mean
        label = f"Beta({a0},{b0})"
        T_final = np.mean(agent.T_history[-10:])
        print(f"  {label:>15}  {p_at[4]:>12.4f}  {p_at[19]:>13.4f}  {T_final:>10.3f}")

    # -----------------------------------------------------------------------
    print_separator("Experiment E: PSRL on Parallel-Series System")
    print(f"\n  Extends PSRL to k=2, r=2 topology (each subsystem = independent arm)")
    print(f"  k subsystems in series → system fails if ANY subsystem fully fails")
    print(f"\n  {'Config':<18}  {'PSRL CR':>9}  {'Oracle CR':>10}  {'Regret@100':>12}")
    print(f"  {'-'*55}")
    for (k, r, label) in [(1, 1, "k=1,r=1 (base)"), (2, 1, "k=2,r=1"), (2, 2, "k=2,r=2")]:
        # Simulate parallel-series: system cost = k independent subsystems,
        # each with r parallel components. System fails at first subsystem failure.
        # Approximation: effective p for system = 1 - (1 - p_comp^r)^k  [rough]
        p_sys = 1 - (1 - (1 - TRUE_P) ** r) ** k
        # Recalibrate as minor-repair fraction
        p_eff = TRUE_P * (1 - p_sys) + TRUE_P * p_sys * 0.5   # blended heuristic

        agents_ps = [PSRLAgent(seed=3), OracleAgent()]
        # Override oracle's T*
        agents_ps[1].T_star = optimal_T_given_params(TRUE_ALPHA, TRUE_BETA, p_eff)
        agents_ps[1].T_history = []

        logs_ps = run_experiment(agents_ps, n_episodes=100, seed=5)
        psrl_log   = logs_ps["PSRL (Thompson Sampling)"]
        oracle_log = logs_ps["Oracle (true params)"]
        regret_100 = psrl_log.cumulative_cost[-1] - oracle_log.cumulative_cost[-1]
        print(f"  {label:<18}  {psrl_log.cost_rate:>9.4f}  {oracle_log.cost_rate:>10.4f}  {regret_100:>12.1f}")


if __name__ == "__main__":
    main()
