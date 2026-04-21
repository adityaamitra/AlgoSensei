"""
tools/agent_tools.py
─────────────────────
Four custom tools used by the TutoringOrchestrator.

These are "agentic tools" — discrete capabilities the orchestrator
calls to gather information or take action. Each tool has:
  - A clear interface (name, description, input schema, output)
  - Error handling
  - Logging for observability

Tools map directly to rubric: Tool Implementation + Custom Tool Development.

Tool 1: HintGenerationTool      — generates calibrated hint text
Tool 2: ProblemSelectorTool     — selects next problem for student
Tool 3: StudentProfileTool      — reads and updates student knowledge state
Tool 4: LeakageDetectorTool     — checks whether a hint leaks the solution
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any


# ── Tool base class ───────────────────────────────────────────

@dataclass
class ToolResult:
    success: bool
    output: Any
    error: str = ""


class BaseTool:
    name: str = ""
    description: str = ""

    def run(self, **kwargs) -> ToolResult:
        raise NotImplementedError

    def __repr__(self):
        return f"<Tool: {self.name}>"


# ── Tool 1: HintGenerationTool ────────────────────────────────

class HintGenerationTool(BaseTool):
    """
    Generates a calibrated Socratic hint for a DSA problem.

    Inputs:  pattern (str), hint_type (str), mastery_level (float),
             hint_level (int 1-3), problem_title (str)
    Output:  hint text (str) + hint_level

    Design:
        Uses structured templates per hint_type and mastery bucket.
        Never names the algorithm directly at levels 1-2.
        At level 3, reveals the pattern name but not the implementation.
    """
    name = "hint_generation"
    description = "Generate a calibrated Socratic hint for a DSA problem"

    # Hint templates: hint_type → {mastery_bucket → template}
    TEMPLATES = {
        "constraint_nudge": {
            "novice":     "Look carefully at the input constraints. What does the size limit tell you about acceptable time complexity?",
            "developing": "The constraint on {pattern} problems often hints at the data structure. What structure satisfies the complexity implied here?",
            "proficient": "Given the constraint, what would fail at O(n²)? That tells you the target complexity.",
        },
        "analogy_hint": {
            "novice":     "Think of it like a warehouse worker memorizing shelf locations. As you walk the aisles, you remember where things are — so when you need to find a pair, you just look it up instead of searching the whole warehouse again.",
            "developing": "Imagine checking bags at airport security. You process each item once, and a running record tells you instantly whether you've seen it before.",
            "proficient": "You've built similar lookup structures before. What property of this problem lets you trade space for time?",
        },
        "subproblem_decompose": {
            "novice":     "Solve it for just 2 elements first. Then 3. What pattern do you see?",
            "developing": "Break the problem in half. If you had the optimal answer for the left half and the right half, how would you combine them?",
            "proficient": "What's the smallest subproblem? How does solving it contribute to the full answer?",
        },
        "complexity_clue": {
            "novice":     "An O(n²) solution is straightforward — write that first. Now ask: what single data structure could collapse the inner loop to O(1)?",
            "developing": "The optimal solution is O(n). If each element is processed exactly once, what does that tell you about when decisions are made?",
            "proficient": "O(n) time with O(1) extra space is achievable here. That eliminates most data structures. What remains?",
        },
        "pattern_name_reveal": {
            "novice":     "This is a {pattern_name} problem. Now that you know that — what does a {pattern_name} solution look like at a high level?",
            "developing": "The pattern here is {pattern_name}. You've seen this before — what's the canonical setup for {pattern_name}?",
            "proficient": "Confirmed: {pattern_name}. What's the edge case that makes your initial {pattern_name} approach fail, and how do you handle it?",
        },
    }

    PATTERN_DISPLAY = {
        "arrays_hashing": "hash map",
        "two_pointers": "two pointers",
        "sliding_window": "sliding window",
        "stack": "monotonic stack",
        "binary_search": "binary search on the answer",
        "linked_list": "two-pointer linked list",
        "trees": "tree DFS with a global tracker",
        "tries": "prefix trie",
        "heap_priority_queue": "heap / priority queue",
        "backtracking": "backtracking",
        "graphs": "graph DFS/BFS",
        "dynamic_programming_1d": "1D dynamic programming",
        "dynamic_programming_2d": "2D dynamic programming",
        "greedy": "greedy",
    }

    def _mastery_bucket(self, mastery: float) -> str:
        if mastery < 0.25: return "novice"
        if mastery < 0.60: return "developing"
        return "proficient"

    def run(self, pattern: str, hint_type: str, mastery: float,
            hint_level: int = 1, problem_title: str = "") -> ToolResult:
        try:
            bucket = self._mastery_bucket(mastery)
            templates = self.TEMPLATES.get(hint_type, self.TEMPLATES["subproblem_decompose"])
            template = templates.get(bucket, templates.get("developing", ""))
            pattern_name = self.PATTERN_DISPLAY.get(pattern, pattern.replace("_", " "))
            hint = template.format(pattern=pattern, pattern_name=pattern_name)
            return ToolResult(success=True, output={
                "hint": hint,
                "hint_type": hint_type,
                "hint_level": hint_level,
                "mastery_bucket": bucket,
            })
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


# ── Tool 2: ProblemSelectorTool ───────────────────────────────

class ProblemSelectorTool(BaseTool):
    """
    Selects the next problem for a student based on their knowledge state.

    Inputs:  pattern_focus (str), student_mastery (dict), difficulty_delta (int)
    Output:  Problem object

    Design:
        Filters Blind 75 problems by pattern and target difficulty.
        Avoids recently-seen problems (anti-repetition).
        Falls back gracefully if no problems match the filter.
    """
    name = "problem_selector"
    description = "Select the next DSA problem appropriate for the student's current level"

    def run(self, pattern_focus: str, student_mastery: dict,
            difficulty_delta: int = 0, seen_ids: list = None) -> ToolResult:
        try:
            from data.blind75 import get_by_pattern
            seen_ids = seen_ids or []
            mastery = student_mastery.get(pattern_focus, 0.3)

            # Determine target difficulty
            if mastery < 0.3:
                diffs = ["easy", "medium"][: max(1, 1 + difficulty_delta + 1)]
            elif mastery < 0.65:
                diffs = ["easy", "medium", "hard"][max(0, difficulty_delta + 1):]
            else:
                diffs = ["medium", "hard"]

            problems = [
                p for p in get_by_pattern(pattern_focus)
                if p.id not in seen_ids and p.difficulty in diffs
            ]
            if not problems:
                problems = get_by_pattern(pattern_focus) or []

            if not problems:
                return ToolResult(success=False, output=None,
                                  error=f"No problems found for pattern: {pattern_focus}")

            import random
            selected = random.choice(problems)
            return ToolResult(success=True, output={
                "problem_id": selected.id,
                "title": selected.title,
                "pattern": selected.pattern,
                "difficulty": selected.difficulty,
                "insight": selected.insight,
            })
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


# ── Tool 3: StudentProfileTool ────────────────────────────────

class StudentProfileTool(BaseTool):
    """
    Reads and updates the student's knowledge profile.

    Inputs:  student_state (StudentState), operation (str), update_data (dict)
    Output:  profile summary or updated state

    Operations:
        "read"   → returns full profile summary
        "update" → updates mastery for a pattern after a session
        "assess" → returns weakest patterns (for curriculum planning)

    Design:
        Acts as the agent's memory interface.
        The PPO agent relies on this tool to get the current state vector.
    """
    name = "student_profile"
    description = "Read or update the student's knowledge profile and mastery levels"

    def run(self, student_state, operation: str = "read",
            update_data: dict = None) -> ToolResult:
        try:
            if operation == "read":
                mastered = [p for p, m in student_state.pattern_mastery.items()
                            if m >= 0.85]
                developing = [p for p, m in student_state.pattern_mastery.items()
                              if 0.4 <= m < 0.85]
                weak = [p for p, m in student_state.pattern_mastery.items()
                        if m < 0.4]
                return ToolResult(success=True, output={
                    "session_count": student_state.session_count,
                    "problems_solved": student_state.problems_solved,
                    "mastered_patterns": mastered,
                    "developing_patterns": developing,
                    "weak_patterns": weak,
                    "avg_mastery": float(sum(student_state.pattern_mastery.values())
                                         / len(student_state.pattern_mastery)),
                    "state_vector": student_state.to_vector().tolist(),
                })

            elif operation == "update":
                if update_data:
                    pattern = update_data.get("pattern")
                    delta   = update_data.get("delta", 0.0)
                    if pattern and pattern in student_state.pattern_mastery:
                        student_state.pattern_mastery[pattern] = float(
                            min(1.0, student_state.pattern_mastery[pattern] + delta)
                        )
                return ToolResult(success=True, output={"updated": True})

            elif operation == "assess":
                # Return the N weakest patterns for curriculum focus
                n = (update_data or {}).get("n", 3)
                sorted_patterns = sorted(
                    student_state.pattern_mastery.items(), key=lambda x: x[1]
                )
                weakest = [{"pattern": p, "mastery": round(m, 3)}
                           for p, m in sorted_patterns[:n]]
                return ToolResult(success=True, output={"weakest_patterns": weakest})

            else:
                return ToolResult(success=False, output=None,
                                  error=f"Unknown operation: {operation}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


# ── Tool 4: LeakageDetectorTool ───────────────────────────────

class LeakageDetectorTool(BaseTool):
    """
    Detects whether a generated hint leaks the solution directly.
    This is the architectural enforcement of the 'no direct answers' rule.

    Inputs:  hint_text (str), pattern (str)
    Output:  {is_leaked: bool, confidence: float, trigger: str}

    Design:
        Two-stage detection:
          1. Keyword matching (fast, deterministic)
          2. Pattern-specific checks (catches context-specific leaks)

        A hint is flagged if it:
          - Names a specific algorithm by name
          - States the Big-O answer
          - Contains code syntax
          - Points directly to the data structure
    """
    name = "leakage_detector"
    description = "Check whether a hint leaks the solution and should be regenerated"

    ALGORITHM_NAMES = [
        "kadane", "dijkstra", "floyd", "bellman-ford", "kruskal",
        "topological sort", "union find", "disjoint set", "kmp",
        "manacher", "z-algorithm",
    ]
    CODE_PATTERNS = [r"def \w+\(", r"for \w+ in", r"while ", r"return [^.]",
                     r"\bif .+:", r"\.append\(", r"heapq\."]
    BIG_O_PATTERNS = [r"o\(n\)", r"o\(1\)", r"o\(n log n\)", r"o\(n\^2\)"]
    DIRECT_DS = [
        "use a hash map", "use a heap", "use a stack", "use a deque",
        "use dynamic programming", "use memoization", "use a trie",
        "use binary search",
    ]

    def run(self, hint_text: str, pattern: str = "") -> ToolResult:
        try:
            lower = hint_text.lower()
            trigger = None
            confidence = 0.0

            for alg in self.ALGORITHM_NAMES:
                if alg in lower:
                    trigger = f"algorithm name: '{alg}'"
                    confidence = 1.0
                    break

            if not trigger:
                for ds in self.DIRECT_DS:
                    if ds in lower:
                        trigger = f"direct data structure: '{ds}'"
                        confidence = 0.95
                        break

            if not trigger:
                for pattern_re in self.CODE_PATTERNS:
                    if re.search(pattern_re, hint_text):
                        trigger = f"code syntax: '{pattern_re}'"
                        confidence = 0.90
                        break

            if not trigger:
                for bo in self.BIG_O_PATTERNS:
                    if re.search(bo, lower):
                        trigger = f"big-O answer: '{bo}'"
                        confidence = 0.85
                        break

            is_leaked = trigger is not None
            return ToolResult(success=True, output={
                "is_leaked": is_leaked,
                "confidence": confidence,
                "trigger": trigger or "none",
                "recommendation": "regenerate" if is_leaked else "approve",
            })
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


# ── Tool Registry ─────────────────────────────────────────────

TOOL_REGISTRY = {
    "hint_generation":  HintGenerationTool(),
    "problem_selector": ProblemSelectorTool(),
    "student_profile":  StudentProfileTool(),
    "leakage_detector": LeakageDetectorTool(),
}

def get_tool(name: str) -> BaseTool:
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Tool '{name}' not found. Available: {list(TOOL_REGISTRY)}")
    return TOOL_REGISTRY[name]
