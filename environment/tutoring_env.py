"""
environment/tutoring_env.py
────────────────────────────
The DSA Tutoring Environment — a gym-style RL environment.

This is the core simulation that both RL agents interact with.
It models a student working through Blind 75 problems.

The environment operates at TWO timescales:
  - MICRO: one hint interaction (used by the UCB Bandit)
  - MACRO: one full tutoring session (used by the PPO Agent)
"""

from __future__ import annotations
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import yaml, os

# ── Load config ───────────────────────────────────────────────
_cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(_cfg_path) as f:
    CFG = yaml.safe_load(f)

PATTERNS     = CFG["dsa_patterns"]       # 14 patterns
HINT_TYPES   = CFG["hint_types"]         # 5 hint types
MASTERY_THR  = CFG["simulator"]["mastery_threshold"]
ABANDON_THR  = CFG["simulator"]["abandon_threshold"]
DECAY        = CFG["simulator"]["knowledge_decay"]
NOISE        = CFG["simulator"]["noise"]


# ── Student Knowledge State ───────────────────────────────────

@dataclass
class StudentState:
    """
    Tracks what a student knows and how they are performing.
    Serves as the RL state for the PPO curriculum agent.
    """
    pattern_mastery: dict = field(default_factory=dict)
    session_count: int = 0
    total_hints: int = 0
    consecutive_failures: int = 0
    problems_solved: int = 0

    def __post_init__(self):
        rng = np.random.default_rng(42)
        for p in PATTERNS:
            if p not in self.pattern_mastery:
                self.pattern_mastery[p] = float(rng.uniform(0.0, 0.35))

    def to_vector(self) -> np.ndarray:
        """Convert to 16-dim feature vector for PPO input."""
        mastery = np.array([self.pattern_mastery[p] for p in PATTERNS], dtype=np.float32)
        extras = np.array([
            min(self.session_count / 30.0, 1.0),
            min(self.total_hints / 150.0, 1.0),
            float(np.mean(mastery)),
            float(np.std(mastery) if np.std(mastery) > 0 else 0.0),
            # solved/attempted ratio (approx)
            min(self.problems_solved / max(self.session_count * 3, 1), 1.0),
            min(self.consecutive_failures / 5.0, 1.0),
        ], dtype=np.float32)
        # Use first 10 mastery dims + 6 extras = 16 total
        return np.concatenate([mastery[:10], extras])

    def knows(self, pattern: str) -> bool:
        return self.pattern_mastery.get(pattern, 0.0) >= MASTERY_THR


# ── Micro Environment: Hint-Level ─────────────────────────────

class HintEnvironment:
    """
    Single-step environment for the UCB Bandit.
    One step = one hint given to a student on one problem.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def step(
        self,
        pattern: str,
        mastery: float,
        hint_type: str,
        is_wrong_direction: bool = False,
    ) -> tuple[float, bool, bool, bool]:
        """
        Simulate student response to a hint.

        Returns: (reward, made_progress, solved, abandoned)
        """
        if is_wrong_direction:
            return -1.5, False, False, False

        quality = self._hint_quality(hint_type, mastery)
        quality += float(self.rng.normal(0, NOISE))
        quality = float(np.clip(quality, 0.0, 1.0))

        progress_prob = 0.35 * mastery + 0.65 * quality
        made_progress = bool(self.rng.random() < progress_prob)
        solved        = bool(self.rng.random() < progress_prob * (0.3 + 0.7 * mastery))
        re_hint       = (not made_progress) and bool(self.rng.random() < 0.7)
        abandoned     = False  # handled at session level

        reward = 0.0
        if made_progress: reward += CFG["bandit"]["reward"]["progress"]
        if solved:        reward += CFG["bandit"]["reward"]["solved"]
        if re_hint:       reward += CFG["bandit"]["reward"]["re_hint"]

        return reward, made_progress, solved, abandoned

    def _hint_quality(self, hint_type: str, mastery: float) -> float:
        """
        Pedagogical model: which hints work at which mastery levels.
        This is the ground truth the bandit must DISCOVER through experience.
        """
        quality_map = {
            # (base, mastery_scaling)
            # Differences are deliberately pronounced so UCB can discover the
            # optimal hint type faster — making learning curves clearly visible.
            "constraint_nudge":   (0.25, 0.60),   # strong for proficient students
            "analogy_hint":       (0.70, -0.35),  # strong for novices, weak for experts
            "subproblem_decompose":(0.45,  0.15),  # moderate across all levels
            "complexity_clue":    (0.10,  0.80),  # near useless for novices, great for experts
            "pattern_name_reveal":(0.38,  0.00),  # flat, mediocre — never optimal
        }
        base, scale = quality_map.get(hint_type, (0.3, 0.2))
        return float(np.clip(base + scale * mastery, 0.0, 1.0))


# ── Macro Environment: Session-Level ─────────────────────────

@dataclass
class SessionResult:
    patterns_mastered: list
    avg_hints_per_problem: float
    student_retained: bool
    wrong_direction_count: int
    problems_attempted: int

class SessionEnvironment:
    """
    Multi-step environment for the PPO Curriculum Agent.
    One episode = one tutoring session (sequence of problems).
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng   = np.random.default_rng(seed)
        self.hint_env = HintEnvironment(seed)

    def new_student(self) -> StudentState:
        s = StudentState()
        # randomize initial mastery with different seed per student
        for p in PATTERNS:
            s.pattern_mastery[p] = float(self.rng.uniform(0.0, 0.35))
        return s

    def run_session(
        self,
        student: StudentState,
        pattern_focus: str,
        difficulty_delta: int,      # -1 easier, 0 same, +1 harder
        start_mode: str,            # "explainer" or "socratic"
        prev_avg_hints: float = 3.0,
    ) -> tuple[float, SessionResult]:
        """
        Simulate a 3-problem tutoring session.
        Returns (total_reward, SessionResult).
        """
        from data.blind75 import get_by_pattern, DIFFICULTY_WEIGHT

        problems = get_by_pattern(pattern_focus)
        if not problems:
            return 0.0, SessionResult([], 3.0, False, 0, 0)

        # Filter by difficulty delta
        target_diff = self._target_difficulty(
            student.pattern_mastery.get(pattern_focus, 0.3), difficulty_delta
        )
        pool = [p for p in problems if p.difficulty == target_diff] or problems
        chosen = [self.rng.choice(pool) for _ in range(min(3, len(pool)))]

        mastered_before = {p for p in PATTERNS if student.knows(p)}
        total_hints = 0
        wrong_count = 0
        abandoned   = False

        for prob in chosen:
            student.problems_solved += 0  # track attempts
            hints_this = 0
            solved = False
            max_hints = 3

            while hints_this < max_hints + 2 and not solved:
                mastery = student.pattern_mastery.get(prob.pattern, 0.0)
                # Simulate 8% wrong-direction rate (naive system baseline)
                wrong = bool(self.rng.random() < 0.08)
                if wrong:
                    wrong_count += 1

                hint = self.rng.choice(HINT_TYPES)
                _, progress, solved, _ = self.hint_env.step(
                    prob.pattern, mastery, hint, is_wrong_direction=wrong
                )
                hints_this += 1
                total_hints += 1

                if progress:
                    delta = float(self.rng.uniform(0.04, 0.14))
                    student.pattern_mastery[prob.pattern] = min(
                        1.0, mastery + delta
                    )
                    student.consecutive_failures = 0
                else:
                    student.consecutive_failures += 1

                if student.consecutive_failures >= ABANDON_THR:
                    abandoned = True
                    break

            if solved:
                student.problems_solved += 1
            if abandoned:
                break

        # Knowledge decay on untouched patterns
        for p in PATTERNS:
            if p != pattern_focus:
                student.pattern_mastery[p] = max(
                    0.0, student.pattern_mastery[p] - DECAY
                )

        student.session_count += 1
        student.total_hints += total_hints

        mastered_after = {p for p in PATTERNS if student.knows(p)}
        newly_mastered = list(mastered_after - mastered_before)
        avg_hints = total_hints / max(len(chosen), 1)

        # ── PPO reward ────────────────────────────────────────
        rw = CFG["ppo"]["reward"]
        reward = 0.0
        reward += len(newly_mastered)             * rw["mastered"]
        reward += max(0, prev_avg_hints - avg_hints) * rw["hint_reduction"]
        if not abandoned:
            reward += rw["retention"]
        reward += wrong_count                     * rw["wrong_direction"]

        return reward, SessionResult(
            patterns_mastered=newly_mastered,
            avg_hints_per_problem=avg_hints,
            student_retained=not abandoned,
            wrong_direction_count=wrong_count,
            problems_attempted=len(chosen),
        )

    def _target_difficulty(self, mastery: float, delta: int) -> str:
        if mastery < 0.3:
            base = ["easy", "easy", "medium"][delta + 1]
        elif mastery < 0.6:
            base = ["easy", "medium", "hard"][delta + 1]
        else:
            base = ["medium", "hard", "hard"][delta + 1]
        return base
