"""
main.py — AlgoSensei RL Entry Point

Usage:
    python main.py --mode test          # smoke test
    python main.py --mode train_bandit  # train UCB bandit only
    python main.py --mode train_ppo     # train PPO agent only
    python main.py --mode train_all     # train both
    python main.py --mode evaluate      # full evaluation + plots
    python main.py --mode demo          # quick demo (fewer episodes)
"""

import argparse, logging, sys, os
sys.path.insert(0, os.path.dirname(__file__))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("algosensei")

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass


def smoke_test():
    import numpy as np
    from data.blind75 import BLIND75, get_patterns
    from environment.tutoring_env import HintEnvironment, SessionEnvironment
    from rl.bandit.ucb_bandit import UCBContextualBandit, encode_context
    from rl.ppo.ppo_agent import decode_action, ACTION_DIM
    from tools.agent_tools import TOOL_REGISTRY
    from baselines.baselines import RandomPolicy, RuleBasedPolicy
    from evaluation.evaluator import improvement_ratio

    log.info("Running smoke test...")

    assert len(BLIND75) >= 75,                  "Blind 75 registry incomplete"
    assert len(get_patterns()) == 14,           "Should have 14 patterns"
    log.info(f"  ✓ Blind 75: {len(BLIND75)} problems, 14 patterns")

    env = HintEnvironment(seed=42)
    r, prog, _, _ = env.step("arrays_hashing", 0.3, "analogy_hint")
    assert isinstance(r, float),                "Reward must be float"
    log.info(f"  ✓ HintEnvironment: reward={r:.2f}, progress={prog}")

    bandit = UCBContextualBandit()
    ctx = encode_context("trees", 0.2)
    h   = bandit.select(ctx)
    bandit.update(ctx, h, 1.0)
    assert h in bandit._Q[ctx],                 "Hint must be in Q-table"
    log.info(f"  ✓ UCB Bandit: selected '{h}'")

    assert ACTION_DIM == 42,                    "PPO action dim should be 42"
    pat, delta = decode_action(0)
    assert pat in get_patterns(),               "Decoded pattern invalid"
    log.info(f"  ✓ PPO action space: {ACTION_DIM} actions, decode(0)=({pat},{delta})")

    assert len(TOOL_REGISTRY) == 4,             "Should have 4 tools"
    for name, tool in TOOL_REGISTRY.items():
        assert hasattr(tool, "run"),            f"Tool {name} missing run()"
    log.info(f"  ✓ Tools: {list(TOOL_REGISTRY.keys())}")

    # Test each tool
    from tools.agent_tools import get_tool
    from environment.tutoring_env import StudentState

    r1 = get_tool("hint_generation").run(
        pattern="arrays_hashing", hint_type="analogy_hint",
        mastery=0.3, hint_level=1, problem_title="Two Sum"
    )
    assert r1.success, f"HintGenerationTool failed: {r1.error}"
    log.info(f"  ✓ HintGenerationTool: '{r1.output['hint'][:60]}...'")

    r2 = get_tool("problem_selector").run(
        pattern_focus="trees", student_mastery={"trees": 0.4}
    )
    assert r2.success, f"ProblemSelectorTool failed: {r2.error}"
    log.info(f"  ✓ ProblemSelectorTool: selected '{r2.output['title']}'")

    student = StudentState()
    r3 = get_tool("student_profile").run(student_state=student, operation="read")
    assert r3.success, f"StudentProfileTool failed: {r3.error}"
    log.info(f"  ✓ StudentProfileTool: {len(r3.output['weak_patterns'])} weak patterns")

    r4 = get_tool("leakage_detector").run(hint_text="Use Kadane's algorithm here.")
    assert r4.success and r4.output["is_leaked"], "LeakageDetectorTool should flag 'Kadane'"
    r5 = get_tool("leakage_detector").run(hint_text="What do you notice about elements seen before?")
    assert r5.success and not r5.output["is_leaked"], "Clean hint should not be flagged"
    log.info(f"  ✓ LeakageDetectorTool: flagged leaked, passed clean")

    log.info("\n  🎉 All smoke tests passed.")
    return True


def train_bandit(n_episodes=10_000, seed=42):
    from rl.bandit.ucb_bandit import train_bandit as _train
    os.makedirs("checkpoints", exist_ok=True)
    log.info(f"Training UCB bandit ({n_episodes} episodes)...")
    bandit = _train(n_episodes=n_episodes, seed=seed, verbose=True)
    bandit.save("checkpoints/bandit.json")
    log.info("Saved: checkpoints/bandit.json")
    return bandit


def train_ppo(n_episodes=3_000, seed=42):
    try:
        import torch
    except ImportError:
        log.error("PyTorch not installed. Run: pip install torch")
        sys.exit(1)
    from rl.ppo.ppo_agent import train_ppo as _train
    os.makedirs("checkpoints", exist_ok=True)
    log.info(f"Training PPO agent ({n_episodes} episodes)...")
    agent = _train(n_episodes=n_episodes, seed=seed, verbose=True)
    agent.save("checkpoints/ppo_agent.pt")
    log.info("Saved: checkpoints/ppo_agent.pt")
    return agent


def run_evaluation(bandit, ppo_agent=None):
    from evaluation.evaluator import (
        evaluate_bandit, evaluate_ppo, run_multi_seed_bandit, plot_all
    )
    import json

    os.makedirs("results", exist_ok=True)
    log.info("Running evaluation suite...")

    log.info("  Evaluating bandit vs baselines (1000 episodes)...")
    bandit_eval = evaluate_bandit(bandit, n_episodes=1000)

    log.info("  Running 5-seed statistical validation...")
    multi_seed = run_multi_seed_bandit(n_seeds=5, n_train=5_000, n_eval=500)

    ppo_eval = {}
    if ppo_agent:
        log.info("  Evaluating PPO vs baselines...")
        ppo_eval = evaluate_ppo(ppo_agent, n_episodes=200)

    log.info("  Generating figures...")
    plot_all(bandit, ppo_agent, bandit_eval, ppo_eval, multi_seed, "results/")

    report = {
        "bandit_eval": bandit_eval,
        "ppo_eval": ppo_eval,
        "multi_seed": multi_seed,
        "bandit_improvement_ratio": multi_seed["improvement_ratio"]["mean"],
    }
    with open("results/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log.info("Saved: results/evaluation_report.json")

    log.info("\n── Evaluation Summary ──────────────────")
    log.info(f"  UCB Bandit mean reward:  {bandit_eval['UCB (trained)']['mean']:.4f}")
    log.info(f"  Random baseline:         {bandit_eval['Random']['mean']:.4f}")
    log.info(f"  Rule-based baseline:     {bandit_eval['Rule-based']['mean']:.4f}")
    ir = multi_seed['improvement_ratio']
    log.info(f"  Improvement ratio:       {ir['mean']:.3f} ± {ir['std']:.3f}")
    if ppo_eval:
        log.info(f"  PPO mean episode reward: {ppo_eval['PPO (trained)']['mean']:.4f}")
        log.info(f"  PPO improvement over random: {ppo_eval['PPO (trained)']['improvement_over_random']:.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=[
        "test", "train_bandit", "train_ppo", "train_all", "evaluate", "demo"
    ], default="test")
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "test":
        smoke_test()

    elif args.mode == "train_bandit":
        train_bandit(n_episodes=args.episodes or 10_000)

    elif args.mode == "train_ppo":
        train_ppo(n_episodes=args.episodes or 3_000)

    elif args.mode == "train_all":
        bandit = train_bandit(n_episodes=args.episodes or 10_000)
        agent  = train_ppo(n_episodes=args.episodes or 3_000)
        run_evaluation(bandit, agent)

    elif args.mode == "evaluate":
        from rl.bandit.ucb_bandit import UCBContextualBandit
        bandit = UCBContextualBandit.load("checkpoints/bandit.json")
        ppo    = None
        if os.path.exists("checkpoints/ppo_agent.pt"):
            from rl.ppo.ppo_agent import PPOCurriculumAgent
            ppo = PPOCurriculumAgent(); ppo.load("checkpoints/ppo_agent.pt")
        run_evaluation(bandit, ppo)

    elif args.mode == "demo":
        log.info("Demo mode: 2000 bandit + 500 PPO episodes")
        bandit = train_bandit(n_episodes=2_000)
        try:
            agent = train_ppo(n_episodes=500)
        except (ImportError, SystemExit):
            agent = None
            log.info("PPO skipped (torch not available)")
        run_evaluation(bandit, agent)


if __name__ == "__main__":
    main()
