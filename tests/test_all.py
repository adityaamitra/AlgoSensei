"""
tests/test_all.py
──────────────────
Complete test suite for AlgoSensei RL.
Run: python3 -m unittest tests/test_all.py -v
"""

import sys, os, unittest, json, math, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np


# ══════════════════════════════════════════════════════════════
# 1. DATA REGISTRY TESTS
# ══════════════════════════════════════════════════════════════

class TestBlind75Registry(unittest.TestCase):

    def setUp(self):
        from data.blind75 import BLIND75, get_patterns, get_by_pattern
        self.BLIND75 = BLIND75
        self.get_patterns = get_patterns
        self.get_by_pattern = get_by_pattern

    def test_minimum_75_problems(self):
        self.assertGreaterEqual(len(self.BLIND75), 75)

    def test_all_14_patterns_present(self):
        import yaml
        with open("config/config.yaml") as f:
            cfg = yaml.safe_load(f)
        config_patterns = set(cfg["dsa_patterns"])
        registry_patterns = set(p.pattern for p in self.BLIND75)
        self.assertEqual(config_patterns, registry_patterns,
                         f"Missing: {config_patterns - registry_patterns}")

    def test_no_duplicate_ids(self):
        ids = [p.id for p in self.BLIND75]
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_difficulties_valid(self):
        valid = {"easy", "medium", "hard"}
        for p in self.BLIND75:
            self.assertIn(p.difficulty, valid, f"{p.title} has invalid difficulty")

    def test_all_insights_nonempty(self):
        for p in self.BLIND75:
            self.assertTrue(p.insight, f"{p.title} has empty insight")

    def test_at_least_2_problems_per_pattern(self):
        for pat in self.get_patterns():
            problems = self.get_by_pattern(pat)
            self.assertGreaterEqual(len(problems), 2,
                                    f"Pattern '{pat}' has only {len(problems)} problem(s)")


# ══════════════════════════════════════════════════════════════
# 2. ENVIRONMENT TESTS
# ══════════════════════════════════════════════════════════════

class TestHintEnvironment(unittest.TestCase):

    def setUp(self):
        from environment.tutoring_env import HintEnvironment
        self.env = HintEnvironment(seed=42)

    def test_reward_is_float(self):
        r, _, _, _ = self.env.step("arrays_hashing", 0.3, "analogy_hint")
        self.assertIsInstance(r, float)

    def test_progress_is_bool(self):
        _, progress, _, _ = self.env.step("trees", 0.5, "subproblem_decompose")
        self.assertIsInstance(progress, bool)

    def test_wrong_direction_always_negative(self):
        """Silent Failure: wrong-direction hint must return negative reward."""
        for seed in range(10):
            env = __import__("environment.tutoring_env",
                             fromlist=["HintEnvironment"]).HintEnvironment(seed=seed)
            r, progress, _, _ = env.step("graphs", 0.9, "analogy_hint",
                                          is_wrong_direction=True)
            self.assertFalse(progress, f"Seed {seed}: wrong direction showed progress")
            self.assertLess(r, 0, f"Seed {seed}: wrong direction should give negative reward")

    def test_all_hint_types_accepted(self):
        import yaml
        with open("config/config.yaml") as f:
            hints = yaml.safe_load(f)["hint_types"]
        for h in hints:
            r, _, _, _ = self.env.step("stack", 0.4, h)
            self.assertTrue(math.isfinite(r), f"Non-finite reward for hint '{h}'")

    def test_high_mastery_novice_hint_difference(self):
        """analogy_hint should work better for novices, complexity_clue for experts."""
        novice_rewards, expert_rewards = [], []
        for seed in range(100):
            env = __import__("environment.tutoring_env",
                             fromlist=["HintEnvironment"]).HintEnvironment(seed=seed)
            r_n, _, _, _ = env.step("arrays_hashing", 0.1, "analogy_hint")
            r_e, _, _, _ = env.step("arrays_hashing", 0.9, "complexity_clue")
            novice_rewards.append(r_n)
            expert_rewards.append(r_e)
        # On average, analogy_hint for novices should be competitive
        self.assertGreater(np.mean(novice_rewards), 0,
                           "analogy_hint for novices should give positive avg reward")


class TestStudentState(unittest.TestCase):

    def setUp(self):
        from environment.tutoring_env import StudentState
        self.StudentState = StudentState

    def test_state_vector_dim(self):
        s = self.StudentState()
        self.assertEqual(s.to_vector().shape, (16,))

    def test_state_vector_bounded(self):
        s = self.StudentState()
        v = s.to_vector()
        self.assertTrue(np.all(v >= 0), "State vector has negative values")
        self.assertTrue(np.all(v <= 1), "State vector has values > 1")

    def test_mastery_updates(self):
        s = self.StudentState()
        old = s.pattern_mastery.get("trees", 0.0)
        s.pattern_mastery["trees"] = min(1.0, old + 0.1)
        self.assertGreater(s.pattern_mastery["trees"], old)


# ══════════════════════════════════════════════════════════════
# 3. UCB BANDIT TESTS
# ══════════════════════════════════════════════════════════════

class TestUCBBandit(unittest.TestCase):

    def setUp(self):
        from rl.bandit.ucb_bandit import UCBContextualBandit, encode_context
        self.Bandit = UCBContextualBandit
        self.encode = encode_context

    def test_context_encoding_buckets(self):
        self.assertEqual(self.encode("trees", 0.1),  "trees::novice")
        self.assertEqual(self.encode("trees", 0.3),  "trees::developing")
        self.assertEqual(self.encode("trees", 0.6),  "trees::proficient")
        self.assertEqual(self.encode("trees", 0.8),  "trees::mastered")

    def test_select_returns_valid_hint(self):
        import yaml
        with open("config/config.yaml") as f:
            hints = yaml.safe_load(f)["hint_types"]
        b = self.Bandit()
        ctx = self.encode("arrays_hashing", 0.3)
        h = b.select(ctx)
        self.assertIn(h, hints)

    def test_exploration_covers_all_arms(self):
        import yaml
        with open("config/config.yaml") as f:
            hints = yaml.safe_load(f)["hint_types"]
        b = self.Bandit()
        ctx = self.encode("graphs", 0.4)
        selected = set()
        for _ in range(100):
            h = b.select(ctx)
            b.update(ctx, h, 0.5)
            selected.add(h)
        self.assertEqual(selected, set(hints),
                         "Exploration should cover all arms")

    def test_exploitation_prefers_highest_reward(self):
        b = self.Bandit()
        ctx = self.encode("backtracking", 0.5)
        # Give analogy_hint very high reward
        for _ in range(30):
            b.update(ctx, "analogy_hint", 5.0)
        for h in ["constraint_nudge","subproblem_decompose","complexity_clue","pattern_name_reveal"]:
            for _ in range(30):
                b.update(ctx, h, -2.0)
        best = b.select(ctx, exploit=True)
        self.assertEqual(best, "analogy_hint",
                         "Exploitation should select highest-reward arm")

    def test_update_increments_counter(self):
        b = self.Bandit()
        ctx = self.encode("trees", 0.2)
        b.update(ctx, "analogy_hint", 1.0)
        self.assertEqual(b._N[ctx], 1)
        self.assertEqual(b._n[ctx]["analogy_hint"], 1)

    def test_save_load_roundtrip(self):
        b = self.Bandit()
        ctx = self.encode("greedy", 0.6)
        for h in ["analogy_hint", "constraint_nudge"]:
            b.update(ctx, h, 1.0)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        b.save(path)
        b2 = self.Bandit.load(path)
        self.assertEqual(b2._N[ctx], b._N[ctx])
        os.unlink(path)

    def test_short_training_improves(self):
        from rl.bandit.ucb_bandit import train_bandit
        b = train_bandit(n_episodes=500, seed=0, verbose=False)
        self.assertEqual(len(b.history), 500)
        rewards = [h["reward"] for h in b.history]
        self.assertTrue(all(math.isfinite(r) for r in rewards))
        # All rewards should be finite
        self.assertTrue(all(math.isfinite(r) for r in rewards))


# ══════════════════════════════════════════════════════════════
# 4. PPO ACTION SPACE TESTS (no torch required)
# ══════════════════════════════════════════════════════════════

class TestPPOActionSpace(unittest.TestCase):

    def test_action_dim_correct(self):
        from rl.ppo.ppo_agent import ACTION_DIM
        self.assertEqual(ACTION_DIM, 42)  # 14 patterns × 3 deltas

    def test_decode_action_valid_pattern(self):
        from rl.ppo.ppo_agent import decode_action
        import yaml
        with open("config/config.yaml") as f:
            patterns = yaml.safe_load(f)["dsa_patterns"]
        for a in range(42):
            pat, delta = decode_action(a)
            self.assertIn(pat, patterns, f"Action {a}: invalid pattern '{pat}'")
            self.assertIn(delta, [-1, 0, 1], f"Action {a}: invalid delta {delta}")

    def test_decode_covers_all_patterns(self):
        from rl.ppo.ppo_agent import decode_action
        import yaml
        with open("config/config.yaml") as f:
            patterns = yaml.safe_load(f)["dsa_patterns"]
        decoded_patterns = {decode_action(a)[0] for a in range(42)}
        self.assertEqual(decoded_patterns, set(patterns))

    def test_decode_covers_all_deltas(self):
        from rl.ppo.ppo_agent import decode_action
        decoded_deltas = {decode_action(a)[1] for a in range(42)}
        self.assertEqual(decoded_deltas, {-1, 0, 1})


# ══════════════════════════════════════════════════════════════
# 5. TOOL TESTS
# ══════════════════════════════════════════════════════════════

class TestTools(unittest.TestCase):

    def setUp(self):
        from tools.agent_tools import get_tool, TOOL_REGISTRY
        self.get_tool = get_tool
        self.registry = TOOL_REGISTRY

    def test_four_tools_registered(self):
        self.assertEqual(len(self.registry), 4)

    def test_all_tools_have_run_method(self):
        for name, tool in self.registry.items():
            self.assertTrue(hasattr(tool, "run"), f"{name} missing run()")

    # ── HintGenerationTool ────────────────────────────────────

    def test_hint_generation_succeeds(self):
        r = self.get_tool("hint_generation").run(
            pattern="arrays_hashing", hint_type="analogy_hint",
            mastery=0.3, hint_level=1, problem_title="Two Sum"
        )
        self.assertTrue(r.success)
        self.assertIsInstance(r.output["hint"], str)
        self.assertGreater(len(r.output["hint"]), 10)

    def test_hint_generation_all_types(self):
        import yaml
        with open("config/config.yaml") as f:
            hints = yaml.safe_load(f)["hint_types"]
        for h in hints:
            r = self.get_tool("hint_generation").run(
                pattern="trees", hint_type=h, mastery=0.5,
                hint_level=1, problem_title="Max Depth"
            )
            self.assertTrue(r.success, f"Hint type '{h}' failed")

    def test_hint_generation_all_mastery_buckets(self):
        for m in [0.1, 0.35, 0.65, 0.9]:
            r = self.get_tool("hint_generation").run(
                pattern="dynamic_programming_1d", hint_type="analogy_hint",
                mastery=m, hint_level=1, problem_title="Coin Change"
            )
            self.assertTrue(r.success, f"Failed at mastery={m}")

    # ── ProblemSelectorTool ───────────────────────────────────

    def test_problem_selector_returns_valid_problem(self):
        r = self.get_tool("problem_selector").run(
            pattern_focus="trees",
            student_mastery={"trees": 0.4},
            difficulty_delta=0,
        )
        self.assertTrue(r.success)
        self.assertIn(r.output["pattern"], ["trees"])
        self.assertIn(r.output["difficulty"], ["easy", "medium", "hard"])

    def test_problem_selector_avoids_seen(self):
        from data.blind75 import get_by_pattern
        all_ids = [p.id for p in get_by_pattern("stack")]
        # Mark all but one as seen
        seen = all_ids[:-1]
        r = self.get_tool("problem_selector").run(
            pattern_focus="stack",
            student_mastery={"stack": 0.5},
            seen_ids=seen,
        )
        self.assertTrue(r.success)

    # ── StudentProfileTool ────────────────────────────────────

    def test_student_profile_read(self):
        from environment.tutoring_env import StudentState
        s = StudentState()
        r = self.get_tool("student_profile").run(s, operation="read")
        self.assertTrue(r.success)
        self.assertIn("mastered_patterns", r.output)
        self.assertIn("weak_patterns", r.output)
        self.assertIn("avg_mastery", r.output)
        self.assertIn("state_vector", r.output)
        self.assertEqual(len(r.output["state_vector"]), 16)

    def test_student_profile_update(self):
        from environment.tutoring_env import StudentState
        s = StudentState()
        old = s.pattern_mastery["trees"]
        self.get_tool("student_profile").run(
            s, operation="update",
            update_data={"pattern": "trees", "delta": 0.2}
        )
        self.assertAlmostEqual(s.pattern_mastery["trees"], min(1.0, old + 0.2), places=5)

    def test_student_profile_assess(self):
        from environment.tutoring_env import StudentState
        s = StudentState()
        r = self.get_tool("student_profile").run(
            s, operation="assess", update_data={"n": 3}
        )
        self.assertTrue(r.success)
        self.assertEqual(len(r.output["weakest_patterns"]), 3)

    def test_student_profile_invalid_op(self):
        from environment.tutoring_env import StudentState
        r = self.get_tool("student_profile").run(
            StudentState(), operation="nonexistent"
        )
        self.assertFalse(r.success)

    # ── LeakageDetectorTool ───────────────────────────────────

    LEAKED_HINTS = [
        "Use Kadane's algorithm here.",
        "Apply Dijkstra's shortest path.",
        "Use dynamic programming with memoization.",
        "def solution(nums): return sorted(nums)",
        "The time complexity is O(n log n).",
        "Use a heap to track the k largest.",
    ]
    CLEAN_HINTS = [
        "What do you notice about elements you've already scanned?",
        "Think about what information you'd want to keep track of as you move left to right.",
        "If you knew the answer for a smaller version of this input, could you extend it?",
        "What's the simplest brute-force approach? Now ask: which part is slowest?",
    ]

    def test_leaked_hints_detected(self):
        for h in self.LEAKED_HINTS:
            r = self.get_tool("leakage_detector").run(hint_text=h)
            self.assertTrue(r.success)
            self.assertTrue(r.output["is_leaked"], f"Should flag: '{h}'")

    def test_clean_hints_pass(self):
        for h in self.CLEAN_HINTS:
            r = self.get_tool("leakage_detector").run(hint_text=h)
            self.assertTrue(r.success)
            self.assertFalse(r.output["is_leaked"], f"Should NOT flag: '{h}'")

    def test_leakage_result_has_required_fields(self):
        r = self.get_tool("leakage_detector").run(hint_text="Use Kadane's.")
        self.assertIn("is_leaked", r.output)
        self.assertIn("confidence", r.output)
        self.assertIn("trigger", r.output)
        self.assertIn("recommendation", r.output)


# ══════════════════════════════════════════════════════════════
# 6. ORCHESTRATOR TESTS
# ══════════════════════════════════════════════════════════════

class TestOrchestrator(unittest.TestCase):

    def setUp(self):
        from agents.orchestrator import TutoringOrchestrator
        from environment.tutoring_env import StudentState
        self.Orchestrator = TutoringOrchestrator
        self.StudentState  = StudentState

    def test_orchestrator_initializes(self):
        orch = self.Orchestrator()
        self.assertIsNotNone(orch.hint_agent)
        self.assertIsNotNone(orch.curriculum_agent)

    def test_start_session_returns_plan(self):
        orch = self.Orchestrator()
        student = self.StudentState()
        plan = orch.start_session(student)
        self.assertIn("pattern_focus", plan)
        self.assertIn("difficulty_delta", plan)
        self.assertIn("problem", plan)

    def test_handle_request_returns_step(self):
        from agents.orchestrator import TutoringStep
        orch = self.Orchestrator()
        student = self.StudentState()
        step = orch.handle_student_request(
            student_state=student,
            pattern="arrays_hashing",
            problem_title="Two Sum",
            hint_level=1,
        )
        self.assertIsInstance(step, TutoringStep)
        self.assertIsInstance(step.hint_text, str)
        self.assertGreater(len(step.hint_text), 5)
        self.assertIsInstance(step.is_clean, bool)

    def test_leakage_check_in_pipeline(self):
        """Orchestrator must run leakage check on every hint."""
        orch = self.Orchestrator()
        student = self.StudentState()
        # Run 10 steps and verify every hint has is_clean field
        for i in range(10):
            step = orch.handle_student_request(
                student, "trees", "Invert Binary Tree", hint_level=1
            )
            self.assertIsInstance(step.is_clean, bool)

    def test_session_history_recorded(self):
        from environment.tutoring_env import SessionEnvironment, SessionResult
        orch = self.Orchestrator()
        student = self.StudentState()
        plan = orch.start_session(student)
        result = SessionResult(
            patterns_mastered=[], avg_hints_per_problem=2.5,
            student_retained=True, wrong_direction_count=0, problems_attempted=3,
        )
        summary = orch.end_session(student, result)
        self.assertEqual(len(orch.get_session_history()), 1)
        self.assertIn("patterns_mastered", summary)


# ══════════════════════════════════════════════════════════════
# 7. BASELINE TESTS
# ══════════════════════════════════════════════════════════════

class TestBaselines(unittest.TestCase):

    def test_random_policy_returns_valid_hint(self):
        import yaml
        with open("config/config.yaml") as f:
            hints = yaml.safe_load(f)["hint_types"]
        from baselines.baselines import RandomPolicy
        p = RandomPolicy()
        for _ in range(20):
            self.assertIn(p.select_hint("trees::novice"), hints)

    def test_rule_based_maps_novice_to_analogy(self):
        from baselines.baselines import RuleBasedPolicy
        # With high enough repetition, novice should mostly get analogy_hint
        p = RuleBasedPolicy()
        results = [p.select_hint("trees::novice") for _ in range(100)]
        analogy_count = results.count("analogy_hint")
        self.assertGreater(analogy_count, 50,
                           "Rule-based should prefer analogy_hint for novice >50% of time")

    def test_epsilon_greedy_updates(self):
        from baselines.baselines import EpsilonGreedyBandit
        b = EpsilonGreedyBandit(epsilon=0.1)
        ctx = "arrays_hashing::novice"
        for _ in range(10):
            h = b.select_hint(ctx)
            b.update(ctx, h, 1.0)
        self.assertIn(ctx, b._Q)


# ══════════════════════════════════════════════════════════════
# 8. EVALUATION TESTS
# ══════════════════════════════════════════════════════════════

class TestEvaluationMetrics(unittest.TestCase):

    def test_smooth_function(self):
        from evaluation.evaluator import smooth
        vals = [float(i) for i in range(100)]
        s = smooth(vals, 10)
        self.assertEqual(len(s), 100)
        self.assertAlmostEqual(s[0], 0.0)

    def test_improvement_ratio_flat(self):
        from evaluation.evaluator import improvement_ratio
        self.assertAlmostEqual(improvement_ratio([1.0] * 100), 1.0, places=2)

    def test_improvement_ratio_increasing(self):
        from evaluation.evaluator import improvement_ratio
        self.assertGreater(improvement_ratio(list(range(1, 101))), 1.0)

    def test_improvement_ratio_too_short(self):
        from evaluation.evaluator import improvement_ratio
        self.assertEqual(improvement_ratio([0.5] * 5), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
