"""
evaluation/evaluator.py
────────────────────────
Evaluation framework with statistical validation.

Metrics:
  1. Cumulative reward per episode          (learning curve)
  2. Patterns mastered per session          (student outcome)
  3. Avg hints per problem over time        (efficiency — lower is better)
  4. Hint leakage rate                      (quality — lower is better)
  5. Student retention rate                 (engagement)
  6. Policy improvement ratio               (vs baseline)
  7. Learning stability (std over 5 seeds)  (reproducibility)

All metrics reported as mean ± std over N_SEEDS runs.
"""

from __future__ import annotations
import json, os
import numpy as np
import yaml

_cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(_cfg_path) as f:
    CFG = yaml.safe_load(f)


def smooth(values: list[float], window: int = 50) -> list[float]:
    out = []
    for i in range(len(values)):
        s = max(0, i - window + 1)
        out.append(float(np.mean(values[s:i+1])))
    return out


def improvement_ratio(rewards: list[float]) -> float:
    if len(rewards) < 20:
        return 1.0
    n   = max(len(rewards) // 10, 10)
    early = np.mean(rewards[:n])
    late  = np.mean(rewards[-n:])
    if abs(early) < 1e-6:
        return float(late - early)
    return float(late / early)


# ── Bandit evaluation ─────────────────────────────────────────

def evaluate_bandit(bandit, n_episodes: int = 1000, seed: int = 0) -> dict:
    """Evaluate trained bandit vs baselines on held-out episodes."""
    from environment.tutoring_env import HintEnvironment
    from rl.bandit.ucb_bandit import encode_context
    from baselines.baselines import RandomPolicy, RuleBasedPolicy, EpsilonGreedyBandit
    from data.blind75 import BLIND75
    import random as py_random

    rng = np.random.default_rng(seed)
    py_random.seed(seed)
    env = HintEnvironment(seed=seed)

    policies = {
        "UCB (trained)": type("P", (), {"select_hint": lambda self, ctx: bandit.select(ctx, exploit=True)})(),
        "Random":        RandomPolicy(),
        "Rule-based":    RuleBasedPolicy(),
        "ε-greedy":      EpsilonGreedyBandit(),
    }
    results = {name: [] for name in policies}

    for _ in range(n_episodes):
        prob    = py_random.choice(BLIND75)
        mastery = float(rng.uniform(0.0, 1.0))
        ctx     = encode_context(prob.pattern, mastery)

        for name, policy in policies.items():
            hint = policy.select_hint(ctx) if hasattr(policy, "select_hint") else bandit.select(ctx, exploit=True)
            r, _, _, _ = env.step(prob.pattern, mastery, hint)
            results[name].append(r)
            # Update adaptive baselines
            if hasattr(policy, "update"):
                policy.update(ctx, hint, r)

    return {
        name: {
            "mean":   float(np.mean(rs)),
            "std":    float(np.std(rs)),
            "median": float(np.median(rs)),
        }
        for name, rs in results.items()
    }


# ── PPO evaluation ────────────────────────────────────────────

def evaluate_ppo(ppo_agent, n_episodes: int = 200, seed: int = 0) -> dict:
    """Evaluate trained PPO vs baselines on curriculum planning."""
    from environment.tutoring_env import SessionEnvironment
    from rl.ppo.ppo_agent import decode_action
    from baselines.baselines import RandomPolicy, RuleBasedPolicy

    env = SessionEnvironment(seed=seed)

    results = {"PPO (trained)": [], "Random": [], "Rule-based": []}
    random_pol = RandomPolicy()
    rule_pol   = RuleBasedPolicy()

    for _ in range(n_episodes):
        # PPO
        student = env.new_student()
        ep_r = 0.0
        prev = 3.0
        for _ in range(CFG["ppo"]["max_steps"]):
            sv = student.to_vector()
            a, _, _ = ppo_agent.select_action(sv, exploit=True)
            pat, delta = decode_action(a)
            r, res = env.run_session(student, pat, delta, "socratic", prev)
            prev = res.avg_hints_per_problem
            ep_r += r
            if not res.student_retained:
                break
        results["PPO (trained)"].append(ep_r)

        # Random baseline
        student = env.new_student()
        ep_r = 0.0
        prev = 3.0
        for _ in range(CFG["ppo"]["max_steps"]):
            pat, delta = random_pol.select_curriculum(student.to_vector())
            r, res = env.run_session(student, pat, delta, "socratic", prev)
            prev = res.avg_hints_per_problem
            ep_r += r
            if not res.student_retained:
                break
        results["Random"].append(ep_r)

        # Rule-based
        student = env.new_student()
        ep_r = 0.0
        prev = 3.0
        for _ in range(CFG["ppo"]["max_steps"]):
            pat, delta = rule_pol.select_curriculum(student.to_vector())
            r, res = env.run_session(student, pat, delta, "socratic", prev)
            prev = res.avg_hints_per_problem
            ep_r += r
            if not res.student_retained:
                break
        results["Rule-based"].append(ep_r)

    return {
        name: {
            "mean": float(np.mean(rs)),
            "std":  float(np.std(rs)),
            "improvement_over_random": float(
                np.mean(rs) / max(np.mean(results["Random"]), 1e-6)
            ) if name == "PPO (trained)" else 1.0,
        }
        for name, rs in results.items()
    }


# ── Multi-seed statistical validation ────────────────────────

def run_multi_seed_bandit(n_seeds: int = 5, n_train: int = 10_000,
                          n_eval: int = 1000) -> dict:
    """Train bandit from N seeds, report mean ± std for statistical validity."""
    from rl.bandit.ucb_bandit import train_bandit

    all_improve, all_mean_reward = [], []

    for seed in range(n_seeds):
        print(f"  Seed {seed+1}/{n_seeds}...", end=" ", flush=True)
        bandit = train_bandit(n_episodes=n_train, seed=seed, verbose=False)
        rewards = [h["reward"] for h in bandit.history]
        all_improve.append(improvement_ratio(rewards))
        all_mean_reward.append(float(np.mean(rewards[-1000:])))
        print(f"improvement={all_improve[-1]:.3f}")

    return {
        "improvement_ratio": {
            "mean": float(np.mean(all_improve)),
            "std":  float(np.std(all_improve)),
            "values": all_improve,
        },
        "mean_reward_final_1k": {
            "mean": float(np.mean(all_mean_reward)),
            "std":  float(np.std(all_mean_reward)),
        },
    }


# ── Plot generation ───────────────────────────────────────────

def plot_all(bandit, ppo_agent, bandit_eval, ppo_eval,
             multi_seed_results, save_dir: str = "results/") -> None:
    """Generate all 5 report figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    os.makedirs(save_dir, exist_ok=True)
    COLORS = {"UCB (trained)": "#2563EB", "Random": "#9CA3AF",
              "Rule-based": "#F59E0B", "ε-greedy": "#10B981",
              "PPO (trained)": "#7C3AED"}

    # ── Figure 1: UCB Bandit learning curve ───────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 1: UCB Contextual Bandit — Learning Curve", fontsize=13)

    rewards = [h["reward"] for h in bandit.history]
    ep      = list(range(1, len(rewards) + 1))
    axes[0].plot(ep, rewards, alpha=0.15, color=COLORS["UCB (trained)"])
    axes[0].plot(ep, smooth(rewards, 500), color=COLORS["UCB (trained)"],
                 linewidth=2.5, label="UCB (smoothed)")
    axes[0].axhline(np.mean(rewards[:1000]), color="gray", linestyle="--",
                    linewidth=1, label="Initial mean")
    axes[0].set_xlabel("Training Episode"); axes[0].set_ylabel("Reward")
    axes[0].set_title("Training Reward Over Time"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Bar chart: UCB vs baselines
    names  = list(bandit_eval.keys())
    means  = [bandit_eval[n]["mean"] for n in names]
    stds   = [bandit_eval[n]["std"]  for n in names]
    colors = [COLORS.get(n, "#6B7280") for n in names]
    bars   = axes[1].bar(names, means, yerr=stds, capsize=5,
                         color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    axes[1].set_ylabel("Mean Reward per Episode")
    axes[1].set_title("UCB vs Baselines (held-out eval)")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(True, alpha=0.3, axis="y")
    for bar, m in zip(bars, means):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{save_dir}fig1_bandit_learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_dir}fig1_bandit_learning_curve.png")

    # ── Figure 2: PPO learning curve ─────────────────────────
    if ppo_agent and ppo_agent.episode_rewards:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Figure 2: PPO Curriculum Agent — Learning Curve", fontsize=13)

        ep_r = ppo_agent.episode_rewards
        ep   = list(range(1, len(ep_r)+1))
        axes[0].plot(ep, ep_r, alpha=0.15, color=COLORS["PPO (trained)"])
        axes[0].plot(ep, smooth(ep_r, 100), color=COLORS["PPO (trained)"],
                     linewidth=2.5, label="PPO (smoothed)")
        axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Episode Reward")
        axes[0].set_title("PPO Reward Over Training"); axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Policy + value losses
        if ppo_agent.policy_losses:
            axes[1].plot(smooth(ppo_agent.policy_losses, 20), color="#DC2626",
                         linewidth=1.5, label="Policy loss")
            axes[1].plot(smooth(ppo_agent.value_losses, 20), color="#2563EB",
                         linewidth=1.5, label="Value loss")
            axes[1].set_xlabel("Update Step"); axes[1].set_ylabel("Loss")
            axes[1].set_title("Policy & Value Losses"); axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        # PPO vs baselines
        names  = list(ppo_eval.keys())
        means  = [ppo_eval[n]["mean"] for n in names]
        colors = [COLORS.get(n, "#6B7280") for n in names]
        axes[2].bar(names, means, color=colors, alpha=0.85,
                    edgecolor="white", linewidth=0.5)
        axes[2].set_ylabel("Mean Episode Reward")
        axes[2].set_title("PPO vs Baselines"); axes[2].grid(True, alpha=0.3, axis="y")
        axes[2].tick_params(axis="x", rotation=15)

        plt.tight_layout()
        plt.savefig(f"{save_dir}fig2_ppo_learning_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_dir}fig2_ppo_learning_curve.png")

    # ── Figure 3: Multi-seed statistical validation ───────────
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Figure 3: Statistical Validation — 5 Seeds", fontsize=13)
    vals   = multi_seed_results["improvement_ratio"]["values"]
    seeds  = [f"Seed {i}" for i in range(len(vals))]
    bars   = ax.bar(seeds, vals, color=COLORS["UCB (trained)"], alpha=0.8,
                    edgecolor="white", linewidth=0.5)
    mean   = multi_seed_results["improvement_ratio"]["mean"]
    std    = multi_seed_results["improvement_ratio"]["std"]
    ax.axhline(mean, color="black", linewidth=1.5, linestyle="--",
               label=f"Mean = {mean:.3f} ± {std:.3f}")
    ax.axhline(1.0, color="gray", linewidth=1, linestyle=":",
               label="Baseline (ratio = 1.0)")
    ax.set_ylabel("Improvement Ratio (final/initial)")
    ax.set_title("Bandit Improvement Ratio Across 5 Seeds")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{save_dir}fig3_multi_seed_validation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_dir}fig3_multi_seed_validation.png")

    # ── Figure 4: Hint policy heatmap ─────────────────────────
    PATTERNS = CFG["dsa_patterns"]
    buckets  = ["novice", "developing", "proficient", "mastered"]
    hint_idx = {h: i for i, h in enumerate(CFG["hint_types"])}
    matrix   = np.full((len(PATTERNS), len(buckets)), -1, dtype=int)
    policy   = bandit.best_policy()

    for i, pat in enumerate(PATTERNS):
        for j, bk in enumerate(buckets):
            ctx = f"{pat}::{bk}"
            if ctx in policy:
                matrix[i][j] = hint_idx.get(policy[ctx], -1)

    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.cm.get_cmap("Set2", len(CFG["hint_types"]))
    im   = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=len(CFG["hint_types"])-1,
                     aspect="auto")
    ax.set_xticks(range(len(buckets)));     ax.set_xticklabels(buckets)
    ax.set_yticks(range(len(PATTERNS)));   ax.set_yticklabels(PATTERNS, fontsize=8)
    ax.set_xlabel("Student Mastery Level")
    ax.set_title("Figure 4: Learned Hint Policy (UCB Bandit)\nBest Hint Type per Context")
    cbar = plt.colorbar(im, ax=ax, ticks=range(len(CFG["hint_types"])))
    cbar.ax.set_yticklabels(CFG["hint_types"], fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{save_dir}fig4_hint_policy_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_dir}fig4_hint_policy_heatmap.png")

    # ── Figure 5: Before/After comparison ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Figure 5: Before vs After RL Training", fontsize=13)

    # Bandit: reward in first 20% vs last 20%
    rewards  = [h["reward"] for h in bandit.history]
    n20      = len(rewards) // 5
    before_b = rewards[:n20]
    after_b  = rewards[-n20:]
    axes[0].boxplot([before_b, after_b], labels=["Before (first 20%)", "After (last 20%)"],
                    patch_artist=True,
                    boxprops=dict(facecolor="#BFDBFE", color="#2563EB"),
                    medianprops=dict(color="#1D4ED8", linewidth=2))
    axes[0].set_ylabel("Reward per Episode"); axes[0].set_title("UCB Bandit: Reward Distribution")
    axes[0].grid(True, alpha=0.3, axis="y")

    # PPO: episode reward in first 20% vs last 20%
    if ppo_agent and ppo_agent.episode_rewards:
        ep_r   = ppo_agent.episode_rewards
        n20p   = len(ep_r) // 5
        before_p = ep_r[:n20p]
        after_p  = ep_r[-n20p:]
        axes[1].boxplot([before_p, after_p], labels=["Before (first 20%)", "After (last 20%)"],
                        patch_artist=True,
                        boxprops=dict(facecolor="#DDD6FE", color="#7C3AED"),
                        medianprops=dict(color="#6D28D9", linewidth=2))
        axes[1].set_ylabel("Episode Reward"); axes[1].set_title("PPO Agent: Episode Reward Distribution")
        axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{save_dir}fig5_before_after.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_dir}fig5_before_after.png")
