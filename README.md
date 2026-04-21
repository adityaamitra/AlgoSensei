# AlgoSensei

**An adaptive DSA tutoring system that uses reinforcement learning to figure out the right hint for the right student at the right time.**

AlgoSensei works through the Blind 75 — the 75 LeetCode problems that every CS student preparing for technical interviews needs to master. Instead of giving answers, it gives calibrated Socratic hints. Two RL agents decide what to say and what to teach next. The system gets better at both tasks through experience.

---

## The Problem It Solves

Most DSA prep follows the same broken loop: get stuck on a problem, look up the solution, skim it, tell yourself you understand it, move on. You never build the underlying intuition. When you hit a slight variation in an actual interview, you're back to square one.

AlgoSensei breaks that loop. It gives you the minimum hint needed to make progress — not the answer. And it learns over time which kind of hint actually works for you on which type of problem.

---

## How It Works

Two RL agents operate at different decision timescales inside an agentic orchestration framework:

| Agent | Algorithm | Decision | Timescale |
|---|---|---|---|
| `HintAgent` | UCB Contextual Bandit | Which hint type to give | Per hint (seconds) |
| `CurriculumAgent` | PPO ActorCritic | Which pattern to focus on next | Per session (minutes) |

**Why two separate agents?** Hint selection is a bandit problem — the reward is immediate and there's no long-horizon planning needed. Curriculum planning is a sequential decision problem — the right problem today shapes what you can learn next week. These are fundamentally different RL problems that require different algorithms. Using one agent for both would be either underpowered or unnecessarily complex.

The orchestrator coordinates both agents through four custom tools and maintains student state across sessions.

---

## Architecture

```
Student request
       │
       ▼
┌─────────────────────────────────────┐
│         TutoringOrchestrator         │
│                                      │
│  1. StudentProfileTool   (memory)    │
│  2. CurriculumAgent ──────────────► session plan
│  3. ProblemSelectorTool ──────────► next problem
│  4. HintAgent ────────────────────► hint type
│  5. HintGenerationTool ───────────► hint text
│  6. LeakageDetectorTool ──────────► safety check
│                                      │
└─────────────────────────────────────┘
       │
       ▼
  Hint delivered to student

         ┌──────────────────────────┐
         │     StudentSimulator      │  ← trains both agents offline
         │  probabilistic student    │     (no real users needed)
         │  model · 14 DSA patterns  │
         └──────────────────────────┘
```

---

## RL Implementation

### Module 1 — UCB Contextual Bandit (hint selection)

At each tutoring step, the `HintAgent` selects one of 5 hint types based on the student's current context.

$$\text{UCB}(a, c) = \hat{Q}(a,c) + C \cdot \sqrt{\frac{\ln N_c}{n(a,c)}}$$

- **Context:** `(DSA pattern, mastery bucket)` → 14 patterns × 4 buckets = **56 contexts**
- **Arms:** `constraint_nudge` · `analogy_hint` · `subproblem_decompose` · `complexity_clue` · `pattern_name_reveal`
- **What it learned:** analogy hints work best for novices, complexity clues for experts — discovered autonomously from reward signals, not programmed in

### Module 2 — PPO Curriculum Agent (session planning)

At the start of each session, the `CurriculumAgent` selects which DSA pattern to focus on and at what difficulty level.

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1{-}\varepsilon, 1{+}\varepsilon)\hat{A}_t\right)\right]$$

- **State:** 16-dim `StudentKnowledgeState` vector (mastery scores + session stats)
- **Actions:** 42 discrete = 14 patterns × 3 difficulty deltas (−1, 0, +1)
- **Network:** `16 → 128 → 64 → [policy head (42) | value head (1)]`
- **GAE:** λ=0.95, γ=0.99

---

## Results

### UCB Bandit — held-out evaluation vs 3 baselines

| Policy | Mean reward | vs. random |
|---|---|---|
| UCB (trained) | **1.3005** | **+16.6%** |
| Rule-based (noisy) | 1.2275 | +10.0% |
| ε-greedy | 1.2065 | +8.2% |
| Random | 1.1155 | — |

UCB beats the rule-based baseline, which has human-encoded domain knowledge. That the bandit exceeds a human-designed heuristic using only reward signals is the clearest evidence of genuine learning.

### Statistical validation — 5 independent seeds

| Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± std |
|---|---|---|---|---|---|
| 1.038 | 1.064 | 1.075 | 1.071 | 1.000 | **1.050 ± 0.028** |

All 5 seeds ≥ 1.0. Results are stable and reproducible.

### PPO Curriculum Agent

| Policy | Mean episode reward | vs. random |
|---|---|---|
| PPO (trained) | **2.1317** | **+39.1%** |
| Random | 1.5325 | — |
| Rule-based | 0.9800 | −36.0% |

The greedy rule-based curriculum (always focus on weakest pattern) actively underperforms random — it ignores student fatigue and triggers abandonment. PPO learns to avoid this.

---

## Agentic Tools

| Tool | Purpose |
|---|---|
| `HintGenerationTool` | Generates Socratic hint text matched to mastery level and hint type |
| `ProblemSelectorTool` | Selects next problem filtered by pattern, mastery, and difficulty delta |
| `StudentProfileTool` | Agent memory interface — reads and updates knowledge state |
| `LeakageDetectorTool` | Blocks hints that contain algorithm names, Big-O answers, or code syntax |

`LeakageDetectorTool` is the safety gate. It runs on every generated hint before delivery. A tutoring system that gives direct answers trains passive learning — this tool enforces the Socratic constraint architecturally, not through prompting.

---

## Quick Start

```bash
# Install dependencies (no external APIs needed)
pip install -r requirements.txt    # numpy, torch, matplotlib, pyyaml

# Verify everything works
python main.py --mode test         # 9 smoke checks — all green

# Train both RL agents (~10 min total)
python main.py --mode train_all    # bandit (10k eps) + PPO (3k eps) + evaluation

# Individual steps
python main.py --mode train_bandit # UCB bandit only (~5 seconds)
python main.py --mode train_ppo    # PPO agent only (~5 min, requires torch)
python main.py --mode evaluate     # run evaluation from saved checkpoints
```

### Jupyter notebook

```bash
cd notebooks/
jupyter notebook train_and_evaluate.ipynb
```

The notebook walks through every section: simulator validation, bandit training, PPO training, evaluation metrics, and all 5 report figures.

### Interactive demo

```bash
streamlit run demo/app.py --server.fileWatcherType none
```

Four screens: live tutoring session, before/after comparison, RL policy visualizer (the heatmap), and leakage detector live demo.

---

## Project Structure

```
algosensei_rl/
├── main.py                          ← entry point
├── requirements.txt
├── config/
│   └── config.yaml                  ← all hyperparameters
├── data/
│   └── blind75.py                   ← 95 problems, 14 DSA patterns
├── environment/
│   └── tutoring_env.py              ← RL training environment
├── rl/
│   ├── bandit/ucb_bandit.py         ← RL Module 1: UCB Contextual Bandit
│   └── ppo/ppo_agent.py             ← RL Module 2: PPO Curriculum Agent
├── agents/
│   └── orchestrator.py              ← TutoringOrchestrator + sub-agents
├── tools/
│   └── agent_tools.py               ← 4 custom agentic tools
├── baselines/
│   └── baselines.py                 ← Random, Rule-based, ε-greedy
├── evaluation/
│   └── evaluator.py                 ← metrics, validation, plots
├── tests/
│   └── test_all.py                  ← 51 unit tests
├── notebooks/
│   └── train_and_evaluate.ipynb     ← full training + evaluation notebook
├── demo/
│   └── app.py                       ← Streamlit demo app
├── checkpoints/                     ← saved model weights
└── results/                         ← evaluation figures
```

---

## Tests

```bash
python -m unittest tests/test_all.py -v
# Ran 51 tests — OK
```

Coverage: data registry, hint environment, student state, UCB bandit (select/update/exploit/save/load), PPO action space, all 4 tools, orchestrator pipeline, baselines, evaluation metrics.

---

## Reproducibility

Seeds are set in `config/config.yaml`. Every result in the evaluation report regenerates from scratch with:

```bash
python main.py --mode train_all
```

The 5-seed validation confirms results are not seed-specific.

---

## Built With

- `numpy` — simulator and bandit
- `torch` — PPO ActorCritic network
- `matplotlib` — evaluation figures
- `streamlit` — demo app
- `pyyaml` — configuration

No external APIs. No paid services. Runs entirely locally.

---

## Author

Aditya Mitra
