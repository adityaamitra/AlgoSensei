"""
rl/bandit/ucb_bandit.py
────────────────────────
RL Module 1: UCB Contextual Bandit for Hint Selection
══════════════════════════════════════════════════════

MATHEMATICAL FORMULATION
─────────────────────────
At each tutoring step, select hint arm a in context c:

    UCB(a, c) = Q̂(a,c) + C · √(ln N_c / n(a,c))

where:
    Q̂(a,c)  = empirical mean reward for arm a in context c
    N_c      = total pulls in context c
    n(a,c)   = pulls of arm a in context c
    C = √2   = exploration constant (UCB1 default)

Context encoding: (pattern, mastery_bucket) → 14 × 4 = 56 contexts
Arms: 5 hint types

Regret bound: E[Regret(T)] ≤ Σ_a (8 ln T / Δ_a) + (1 + π²/3) Σ_a Δ_a
where Δ_a is the suboptimality gap for arm a.
"""

from __future__ import annotations
import json, math, os, yaml
from collections import defaultdict
from typing import Optional
import numpy as np

_cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yaml")
with open(_cfg_path) as f:
    CFG = yaml.safe_load(f)

HINT_TYPES = CFG["hint_types"]
UCB_C      = CFG["bandit"]["ucb_c"]
MIN_PULLS  = CFG["bandit"]["min_pulls"]


# ── Context encoding ──────────────────────────────────────────

def encode_context(pattern: str, mastery: float) -> str:
    """Discretize mastery into 4 buckets → context key."""
    if mastery < 0.25:   bucket = "novice"
    elif mastery < 0.50: bucket = "developing"
    elif mastery < 0.75: bucket = "proficient"
    else:                bucket = "mastered"
    return f"{pattern}::{bucket}"


# ── UCB Bandit ────────────────────────────────────────────────

class UCBContextualBandit:
    """
    Contextual UCB bandit — hint type selection per student-problem context.

    Learns WHICH HINT TYPE works best for each (pattern, mastery) combination.
    The learned policy is interpretable: a heat map of best hint per context.
    """

    def __init__(self):
        # Q̂(a,c): cumulative reward / pulls
        self._Q: dict = defaultdict(lambda: {h: 0.0 for h in HINT_TYPES})
        self._n: dict = defaultdict(lambda: {h: 0   for h in HINT_TYPES})
        self._N: dict = defaultdict(int)   # total pulls per context
        self.history: list[dict] = []
        self._t = 0

    def select(self, context: str, exploit: bool = False) -> str:
        """Select hint type for context. exploit=True → greedy (no exploration)."""
        # Exploration phase: pull each arm MIN_PULLS times first
        for h in HINT_TYPES:
            if self._n[context][h] < MIN_PULLS:
                return h

        if exploit:
            return max(HINT_TYPES, key=lambda h: self._Q[context][h])

        # UCB selection
        N = self._N[context]
        scores = {
            h: self._Q[context][h] + UCB_C * math.sqrt(math.log(N) / self._n[context][h])
            for h in HINT_TYPES
        }
        return max(scores, key=scores.__getitem__)

    def update(self, context: str, hint: str, reward: float) -> None:
        """Incremental update: Q̂(a,c) ← Q̂(a,c) + (r - Q̂(a,c)) / n(a,c)"""
        self._n[context][hint] += 1
        self._N[context] += 1
        # Incremental mean update (numerically stable)
        n = self._n[context][hint]
        self._Q[context][hint] += (reward - self._Q[context][hint]) / n
        self._t += 1
        self.history.append({
            "t": self._t, "context": context,
            "hint": hint, "reward": reward,
            "q_value": self._Q[context][hint],
        })

    def best_policy(self) -> dict[str, str]:
        """Return learned best hint per context (for analysis/reporting)."""
        return {
            ctx: max(HINT_TYPES, key=lambda h: self._Q[ctx][h])
            for ctx in self._Q
            if self._N[ctx] > MIN_PULLS * len(HINT_TYPES)
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "Q": {k: v for k, v in self._Q.items()},
                "n": {k: v for k, v in self._n.items()},
                "N": dict(self._N), "t": self._t,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "UCBContextualBandit":
        b = cls()
        with open(path) as f:
            d = json.load(f)
        b._Q = defaultdict(lambda: {h: 0.0 for h in HINT_TYPES}, d["Q"])
        b._n = defaultdict(lambda: {h: 0   for h in HINT_TYPES},
                           {k: {h: int(v) for h, v in vv.items()} for k, vv in d["n"].items()})
        b._N = defaultdict(int, {k: int(v) for k, v in d["N"].items()})
        b._t = d["t"]
        return b


# ── Training ──────────────────────────────────────────────────

def train_bandit(n_episodes: int = 10_000, seed: int = 42,
                 verbose: bool = True) -> UCBContextualBandit:
    """Train bandit on the hint environment. Returns trained bandit."""
    from environment.tutoring_env import HintEnvironment
    from data.blind75 import BLIND75
    import random as py_random

    rng = np.random.default_rng(seed)
    py_random.seed(seed)
    env = HintEnvironment(seed=seed)
    bandit = UCBContextualBandit()
    rw_cfg = CFG["bandit"]["reward"]

    for ep in range(n_episodes):
        prob    = py_random.choice(BLIND75)
        mastery = float(rng.uniform(0.0, 1.0))
        context = encode_context(prob.pattern, mastery)
        hint    = bandit.select(context)

        reward, progress, solved, _ = env.step(prob.pattern, mastery, hint)
        bandit.update(context, hint, reward)

        if verbose and (ep + 1) % 2000 == 0:
            recent = [h["reward"] for h in bandit.history[-2000:]]
            print(f"  Episode {ep+1:6d} | Mean reward (2k): {np.mean(recent):.4f}")

    return bandit
