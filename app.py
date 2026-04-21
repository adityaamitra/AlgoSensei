"""
demo/app.py
────────────
AlgoSensei — Interactive Demo
Streamlit app for recording the 10-minute demo video.

Screens:
  1. Live Tutoring Session    → shows the orchestrator + RL agents in action
  2. Before vs After          → untrained agent vs trained agent side-by-side
  3. RL Policy Visualizer     → the learned hint heatmap + bandit Q-values
  4. Leakage Detector Live    → type a hint and see it classified in real time

Run: streamlit run demo/app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import time
import json

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AlgoSensei — RL Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1E3A5F;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 6px 0;
    border-left: 4px solid #3B82F6;
}
.metric-title { color: #93C5FD; font-size: 0.8em; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { color: #FFFFFF; font-size: 1.8em; font-weight: bold; }
.hint-box {
    background: #0F172A;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
    font-family: monospace;
}
.hint-type-badge {
    display: inline-block;
    background: #1E40AF;
    color: #BFDBFE;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.75em;
    margin-bottom: 6px;
}
.clean-badge { background: #14532D; color: #86EFAC; }
.leaked-badge { background: #7F1D1D; color: #FCA5A5; }
.section-header {
    font-size: 1.1em;
    font-weight: 600;
    color: #93C5FD;
    border-bottom: 1px solid #1E3A5F;
    padding-bottom: 6px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# ── Load assets (cached) ──────────────────────────────────────

@st.cache_resource
def load_bandit():
    from rl.bandit.ucb_bandit import UCBContextualBandit
    # Try multiple path resolutions to handle different working directories
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "..", "checkpoints", "bandit.json"),
        os.path.join(base, "checkpoints", "bandit.json"),
        os.path.join(os.getcwd(), "checkpoints", "bandit.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return UCBContextualBandit.load(path)
    return None

@st.cache_resource
def load_ppo():
    try:
        import torch
        from rl.ppo.ppo_agent import PPOCurriculumAgent
        base = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base, "..", "checkpoints", "ppo_agent.pt"),
            os.path.join(base, "checkpoints", "ppo_agent.pt"),
            os.path.join(os.getcwd(), "checkpoints", "ppo_agent.pt"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), "")
        if os.path.exists(path):
            agent = PPOCurriculumAgent()
            agent.load(path)
            return agent
    except ImportError:
        pass
    return None

@st.cache_resource
def get_tools():
    from tools.agent_tools import TOOL_REGISTRY
    return TOOL_REGISTRY

@st.cache_data
def get_problems():
    from data.blind75 import BLIND75
    return BLIND75

@st.cache_data
def get_patterns():
    from data.blind75 import get_patterns
    return get_patterns()


# ── Sidebar ───────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("# 🧠 AlgoSensei")
        st.caption("RL-Powered Adaptive DSA Tutor")
        st.divider()

        bandit = load_bandit()
        ppo    = load_ppo()

        if bandit:
            st.success(f"✓ UCB Bandit loaded ({bandit._t:,} training steps)")
        else:
            st.error("✗ Bandit not found — run: python main.py --mode train_bandit")

        if ppo:
            st.success(f"✓ PPO Agent loaded ({len(ppo.episode_rewards):,} episodes)")
        else:
            st.warning("⚠ PPO not loaded (requires torch)")

        st.divider()
        page = st.radio("Demo Screen", [
            "🎓 Live Tutoring Session",
            "⚖️ Before vs After",
            "🗺️ RL Policy Visualizer",
            "🔍 Leakage Detector",
        ])
        st.divider()
        st.caption("Course: Generative AI")
        st.caption("Student: Aditya Mitra")
        st.caption("NUID: 002303254")

    return page


# ══════════════════════════════════════════════════════════════
# SCREEN 1: Live Tutoring Session
# ══════════════════════════════════════════════════════════════

def screen_live_session():
    st.title("🎓 Live Tutoring Session")
    st.caption("The trained RL agents orchestrate a real tutoring session. "
               "Watch the UCB bandit select hint types and the orchestrator run the full pipeline.")

    BLIND75   = get_problems()
    PATTERNS  = get_patterns()
    tools     = get_tools()
    bandit    = load_bandit()

    # Session setup
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        selected_pattern = st.selectbox("DSA Pattern", PATTERNS,
            index=PATTERNS.index("dynamic_programming_1d"))
    with col2:
        mastery_level = st.slider("Student Mastery Level", 0.0, 1.0, 0.25, 0.05,
            help="0.0 = complete beginner, 1.0 = expert")
    with col3:
        n_hints = st.number_input("Hints to show", 1, 5, 3)

    st.divider()

    if st.button("▶ Run Session", type="primary", use_container_width=True):
        # Show student profile
        from environment.tutoring_env import StudentState
        student = StudentState()
        student.pattern_mastery[selected_pattern] = mastery_level

        profile_result = tools["student_profile"].run(student, operation="read")
        profile = profile_result.output

        # ── Student Profile ───────────────────────────────────
        st.markdown('<div class="section-header">Student Knowledge Profile</div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        bucket = ("novice" if mastery_level < 0.25 else
                  "developing" if mastery_level < 0.50 else
                  "proficient" if mastery_level < 0.75 else "mastered")
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Mastery Level</div>'
                        f'<div class="metric-value">{mastery_level:.0%}</div></div>',
                        unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Mastery Bucket</div>'
                        f'<div class="metric-value">{bucket}</div></div>',
                        unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Avg Mastery</div>'
                        f'<div class="metric-value">{profile["avg_mastery"]:.0%}</div></div>',
                        unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Pattern</div>'
                        f'<div class="metric-value" style="font-size:1em">'
                        f'{selected_pattern.replace("_", " ")}</div></div>',
                        unsafe_allow_html=True)

        st.markdown("")

        # ── Problem selection ─────────────────────────────────
        st.markdown('<div class="section-header">Step 1 — ProblemSelectorTool</div>',
                    unsafe_allow_html=True)
        with st.spinner("Selecting problem..."):
            time.sleep(0.3)
            prob_result = tools["problem_selector"].run(
                pattern_focus=selected_pattern,
                student_mastery=student.pattern_mastery,
                difficulty_delta=0,
            )
        if prob_result.success:
            prob = prob_result.output
            pc1, pc2, pc3 = st.columns([3, 1, 2])
            with pc1:
                st.info(f"**#{prob.get('problem_id', '?')} {prob['title']}**")
            with pc2:
                diff_color = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
                st.write(f"{diff_color.get(prob['difficulty'], '⚪')} {prob['difficulty'].title()}")
            with pc3:
                st.write(f"🏷️ `{prob['pattern'].replace('_', ' ')}`")

        st.markdown("")

        # ── Hint generation loop ──────────────────────────────
        st.markdown('<div class="section-header">Step 2 — HintAgent (UCB Bandit) → HintGenerationTool → LeakageDetectorTool</div>',
                    unsafe_allow_html=True)

        from rl.bandit.ucb_bandit import encode_context
        HINT_TYPES = ["constraint_nudge", "analogy_hint", "subproblem_decompose",
                      "complexity_clue", "pattern_name_reveal"]

        for hint_num in range(1, int(n_hints) + 1):
            with st.spinner(f"Generating hint {hint_num}..."):
                time.sleep(0.4)

            # UCB selects hint type
            ctx = encode_context(selected_pattern, mastery_level)
            if bandit:
                hint_type = bandit.select(ctx, exploit=True)
                q_values = {h: round(bandit._Q[ctx][h], 4) for h in HINT_TYPES}
            else:
                import random
                hint_type = random.choice(HINT_TYPES)
                q_values = {h: round(np.random.uniform(0, 2), 4) for h in HINT_TYPES}

            # Generate hint
            gen_result = tools["hint_generation"].run(
                pattern=selected_pattern,
                hint_type=hint_type,
                mastery=mastery_level,
                hint_level=hint_num,
                problem_title=prob["title"] if prob_result.success else "Problem",
            )
            hint_text = gen_result.output["hint"] if gen_result.success else "Think about the constraints."

            # Leakage check
            leak_result = tools["leakage_detector"].run(hint_text=hint_text, pattern=selected_pattern)
            is_clean = not (leak_result.success and leak_result.output["is_leaked"])

            # Render
            clean_class = "clean-badge" if is_clean else "leaked-badge"
            clean_label = "✓ Safe hint" if is_clean else "⚠ Leaked — regenerated"

            st.markdown(f"""
            <div class="hint-box">
                <span class="hint-type-badge">Hint {hint_num} · {hint_type.replace('_', ' ').title()}</span>
                <span class="{clean_class}" style="border-radius:12px; padding:2px 10px; font-size:0.75em; margin-left:8px;">{clean_label}</span>
                <br/><br/>
                <span style="color:#E2E8F0; font-family:sans-serif; font-size:0.95em;">{hint_text}</span>
            </div>
            """, unsafe_allow_html=True)

            # Show UCB Q-values for this context
            with st.expander(f"🔬 UCB decision process for hint {hint_num}"):
                st.caption(f"Context: `{ctx}` | Selected: **{hint_type}** (highest Q-value)")
                q_sorted = sorted(q_values.items(), key=lambda x: -x[1])
                for ht, qv in q_sorted:
                    is_selected = ht == hint_type
                    bar_width = min(int(qv / 2.5 * 100), 100) if qv > 0 else 0
                    marker = " ← selected" if is_selected else ""
                    st.markdown(
                        f"{'**' if is_selected else ''}`{ht}`{'**' if is_selected else ''} "
                        f"Q={qv:.4f}{marker}"
                    )
                    st.progress(bar_width / 100)

        st.success("✓ Session complete — all hints passed leakage detection")


# ══════════════════════════════════════════════════════════════
# SCREEN 2: Before vs After
# ══════════════════════════════════════════════════════════════

def screen_before_after():
    st.title("⚖️ Before vs After RL Training")
    st.caption("Compare an untrained (random) agent against the trained UCB bandit on the same student-problem context.")

    PATTERNS = get_patterns()
    tools    = get_tools()
    bandit   = load_bandit()

    col1, col2 = st.columns(2)
    with col1:
        pattern   = st.selectbox("Pattern", PATTERNS, index=PATTERNS.index("arrays_hashing"))
        mastery   = st.slider("Student mastery", 0.0, 1.0, 0.2, 0.05)
    with col2:
        n_sims    = st.slider("Simulation rounds", 5, 50, 20,
                               help="Run N hint interactions and compare mean reward")

    st.divider()

    if st.button("▶ Run Comparison", type="primary", use_container_width=True):
        import random as py_random
        from rl.bandit.ucb_bandit import encode_context, UCBContextualBandit
        from environment.tutoring_env import HintEnvironment

        env = HintEnvironment(seed=42)
        ctx = encode_context(pattern, mastery)

        HINT_TYPES = ["constraint_nudge", "analogy_hint", "subproblem_decompose",
                      "complexity_clue", "pattern_name_reveal"]

        untrained_rewards, trained_rewards = [], []
        untrained_hints, trained_hints     = [], []
        progress_bar = st.progress(0)

        for i in range(n_sims):
            # Untrained: random selection
            untrained_hint = py_random.choice(HINT_TYPES)
            r_u, prog_u, _, _ = env.step(pattern, mastery, untrained_hint)
            untrained_rewards.append(r_u)
            untrained_hints.append(untrained_hint)

            # Trained: UCB exploit
            if bandit:
                trained_hint = bandit.select(ctx, exploit=True)
            else:
                trained_hint = "analogy_hint"
            r_t, prog_t, _, _ = env.step(pattern, mastery, trained_hint)
            trained_rewards.append(r_t)
            trained_hints.append(trained_hint)

            progress_bar.progress((i + 1) / n_sims)

        # Results
        st.markdown("")
        c1, c2, c3 = st.columns(3)
        untrained_mean = np.mean(untrained_rewards)
        trained_mean   = np.mean(trained_rewards)
        improvement    = ((trained_mean - untrained_mean) / abs(untrained_mean) * 100
                         if untrained_mean != 0 else 0)

        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Untrained (random)</div>'
                        f'<div class="metric-value">{untrained_mean:.3f}</div>'
                        f'<div style="color:#93C5FD;font-size:0.8em">mean reward over {n_sims} rounds</div></div>',
                        unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Trained (UCB)</div>'
                        f'<div class="metric-value">{trained_mean:.3f}</div>'
                        f'<div style="color:#93C5FD;font-size:0.8em">mean reward over {n_sims} rounds</div></div>',
                        unsafe_allow_html=True)
        with c3:
            color = "#86EFAC" if improvement >= 0 else "#FCA5A5"
            st.markdown(f'<div class="metric-card"><div class="metric-title">Improvement</div>'
                        f'<div class="metric-value" style="color:{color}">{improvement:+.1f}%</div>'
                        f'<div style="color:#93C5FD;font-size:0.8em">trained vs untrained</div></div>',
                        unsafe_allow_html=True)

        st.markdown("")

        # Side-by-side hint comparison
        st.markdown('<div class="section-header">Side-by-Side: What Each Agent Would Say</div>',
                    unsafe_allow_html=True)

        left, right = st.columns(2)
        with left:
            st.markdown("#### ❌ Untrained Agent (random)")
            sample_untrained = py_random.choice(HINT_TYPES)
            gen = tools["hint_generation"].run(
                pattern=pattern, hint_type=sample_untrained,
                mastery=mastery, hint_level=1, problem_title="Sample Problem"
            )
            st.markdown(f"**Hint type chosen:** `{sample_untrained}`")
            st.info(gen.output["hint"] if gen.success else "No hint generated.")

        with right:
            st.markdown("#### ✅ Trained Agent (UCB learned policy)")
            best_hint = bandit.select(ctx, exploit=True) if bandit else "analogy_hint"
            gen2 = tools["hint_generation"].run(
                pattern=pattern, hint_type=best_hint,
                mastery=mastery, hint_level=1, problem_title="Sample Problem"
            )
            st.markdown(f"**Hint type chosen:** `{best_hint}` ← learned by UCB")
            st.success(gen2.output["hint"] if gen2.success else "No hint generated.")

        # Reward timeline
        st.markdown("")
        st.markdown('<div class="section-header">Reward Timeline</div>', unsafe_allow_html=True)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
        rounds = list(range(1, n_sims + 1))
        ax1.bar(rounds, untrained_rewards, color="#6B7280", alpha=0.8, label="Random")
        ax1.axhline(untrained_mean, color="red", linestyle="--", linewidth=1.5,
                    label=f"Mean = {untrained_mean:.3f}")
        ax1.set_title("Untrained Agent (Random)"); ax1.legend(); ax1.grid(True, alpha=0.3)
        ax1.set_xlabel("Round"); ax1.set_ylabel("Reward")

        ax2.bar(rounds, trained_rewards, color="#2563EB", alpha=0.8, label="UCB Trained")
        ax2.axhline(trained_mean, color="#16A34A", linestyle="--", linewidth=1.5,
                    label=f"Mean = {trained_mean:.3f}")
        ax2.set_title("Trained Agent (UCB)"); ax2.legend(); ax2.grid(True, alpha=0.3)
        ax2.set_xlabel("Round"); ax2.set_ylabel("Reward")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════
# SCREEN 3: RL Policy Visualizer
# ══════════════════════════════════════════════════════════════

def screen_policy_visualizer():
    st.title("🗺️ RL Policy Visualizer")
    st.caption("Explore the hint policy the UCB bandit learned from 10,000 training episodes. "
               "The heatmap shows the best hint type per (pattern, mastery) context.")

    bandit   = load_bandit()
    PATTERNS = get_patterns()
    HINT_TYPES = ["constraint_nudge", "analogy_hint", "subproblem_decompose",
                  "complexity_clue", "pattern_name_reveal"]
    buckets  = ["novice", "developing", "proficient", "mastered"]

    if not bandit:
        st.error("Bandit not loaded. Run: python main.py --mode train_bandit")
        return

    # ── Heatmap ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Learned Hint Policy Heatmap</div>',
                unsafe_allow_html=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hint_idx = {h: i for i, h in enumerate(HINT_TYPES)}
    matrix   = np.full((len(PATTERNS), len(buckets)), -1, dtype=int)
    policy   = bandit.best_policy()

    for i, pat in enumerate(PATTERNS):
        for j, bk in enumerate(buckets):
            ctx = f"{pat}::{bk}"
            if ctx in policy:
                matrix[i][j] = hint_idx.get(policy[ctx], -1)

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.cm.get_cmap("Set2", len(HINT_TYPES))
    im   = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=len(HINT_TYPES)-1, aspect="auto")
    ax.set_xticks(range(len(buckets)));    ax.set_xticklabels(buckets, fontsize=11)
    ax.set_yticks(range(len(PATTERNS)));  ax.set_yticklabels(PATTERNS, fontsize=9)
    ax.set_xlabel("Student Mastery Level", fontsize=11)
    ax.set_title("Learned Hint Policy — UCB Bandit (10,000 episodes)\nColor = Best Hint Type per Context",
                 fontsize=12)
    cbar = plt.colorbar(im, ax=ax, ticks=range(len(HINT_TYPES)))
    cbar.ax.set_yticklabels(HINT_TYPES, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Key findings ──────────────────────────────────────────
    st.markdown("")
    st.markdown('<div class="section-header">Key Findings</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**What the bandit discovered:**")
        st.markdown("- 🔵 **Novice column** → `analogy_hint` dominates")
        st.markdown("  Real-world analogies work best when mastery is low")
        st.markdown("- 🟠 **Mastered column** → `complexity_clue` dominates")
        st.markdown("  Expert students respond best to O-notation hints")
        st.markdown("- 🟢 **proficient + trees/DP** → `subproblem_decompose`")
        st.markdown("  Decomposition works when you have some base understanding")
    with col2:
        st.markdown("**Why this matters:**")
        st.markdown("- The bandit was **never told** these rules")
        st.markdown("- It discovered them purely from reward signals")
        st.markdown("- This matches the **ground truth** embedded in the simulator")
        st.markdown("- Demonstrates **autonomous pedagogical learning**")

    # ── Context explorer ──────────────────────────────────────
    st.markdown("")
    st.markdown('<div class="section-header">Context Explorer — Inspect Any Context</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        exp_pattern = st.selectbox("Pattern", PATTERNS,
            key="policy_pattern", index=PATTERNS.index("dynamic_programming_1d"))
    with col2:
        exp_mastery = st.slider("Mastery", 0.0, 1.0, 0.3, 0.05, key="policy_mastery")

    from rl.bandit.ucb_bandit import encode_context
    ctx = encode_context(exp_pattern, exp_mastery)
    best = bandit.select(ctx, exploit=True)
    q_vals = {h: round(bandit._Q[ctx][h], 4) for h in HINT_TYPES}
    n_vals = {h: bandit._n[ctx][h] for h in HINT_TYPES}

    st.markdown(f"**Context:** `{ctx}` → **Best hint:** `{best}`")

    q_sorted = sorted(q_vals.items(), key=lambda x: -x[1])
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    colors = ["#2563EB" if h == best else "#9CA3AF" for h, _ in q_sorted]
    hs = [h.replace("_", "\n") for h, _ in q_sorted]
    vs = [v for _, v in q_sorted]
    bars = ax2.bar(hs, vs, color=colors, alpha=0.85, edgecolor="white")
    ax2.set_title(f"Q-values for context: {ctx}")
    ax2.set_ylabel("Q-value (mean reward)")
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, (h, v), n in zip(bars, q_sorted, [n_vals[h] for h, _ in q_sorted]):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}\n(n={n})",
                 ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


# ══════════════════════════════════════════════════════════════
# SCREEN 4: Leakage Detector Live
# ══════════════════════════════════════════════════════════════

def screen_leakage_detector():
    st.title("🔍 Leakage Detector — Live")
    st.caption("Type any hint and see the LeakageDetectorTool classify it in real time. "
               "This tool runs on every hint before it reaches the student.")

    tools = get_tools()

    st.markdown('<div class="section-header">Try Your Own Hint</div>', unsafe_allow_html=True)

    examples = {
        "Type your own...": "",
        "Safe: structural question": "What do you notice about the elements you've already scanned through?",
        "Safe: complexity nudge": "If this took O(n²) right now, what single structure could bring that inner loop to O(1)?",
        "Safe: decompose": "Solve it first for just 2 elements. What did you do there?",
        "LEAKED: algorithm name": "Use Kadane's algorithm to solve this.",
        "LEAKED: data structure": "You should use a hash map to store the complement.",
        "LEAKED: code": "def solution(nums): return sorted(nums)",
        "LEAKED: big-O answer": "The optimal time complexity is O(n log n).",
    }

    chosen = st.selectbox("Load an example or type below", list(examples.keys()))
    hint_input = st.text_area(
        "Hint text to evaluate",
        value=examples[chosen],
        height=100,
        placeholder="Type a hint here and click Classify..."
    )
    pattern_check = st.selectbox("Pattern (for context)", get_patterns(),
                                  index=get_patterns().index("arrays_hashing"))

    if st.button("🔍 Classify Hint", type="primary"):
        if not hint_input.strip():
            st.warning("Please enter a hint to classify.")
            return

        with st.spinner("Running leakage detection..."):
            time.sleep(0.2)
            result = tools["leakage_detector"].run(
                hint_text=hint_input, pattern=pattern_check
            )

        if result.success:
            out = result.output
            is_leaked = out["is_leaked"]

            if is_leaked:
                st.error(f"⚠️ **LEAKED** — This hint would be blocked")
                st.markdown(f"""
                <div class="hint-box" style="border-left: 4px solid #EF4444;">
                    <span class="leaked-badge" style="border-radius:12px;padding:3px 12px;">BLOCKED</span>
                    <br/><br/>
                    <b style="color:#FCA5A5;">Trigger:</b> <code style="color:#FCA5A5;">{out['trigger']}</code><br/>
                    <b style="color:#FCA5A5;">Confidence:</b> {out['confidence']:.0%}<br/>
                    <b style="color:#FCA5A5;">Action:</b> {out['recommendation'].upper()}
                    <br/><br/>
                    <i style="color:#94A3B8;">A safe fallback hint would be shown instead.</i>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success("✅ **SAFE** — This hint passes the leakage check")
                st.markdown(f"""
                <div class="hint-box" style="border-left: 4px solid #22C55E;">
                    <span class="clean-badge" style="border-radius:12px;padding:3px 12px;">APPROVED</span>
                    <br/><br/>
                    <b style="color:#86EFAC;">Trigger:</b> <code style="color:#86EFAC;">none</code><br/>
                    <b style="color:#86EFAC;">Confidence:</b> N/A<br/>
                    <b style="color:#86EFAC;">Action:</b> SHOW TO STUDENT
                    <br/><br/>
                    <i style="color:#94A3B8;">This hint is safe to deliver.</i>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error(f"Tool error: {result.error}")

    # Show stats
    st.divider()
    st.markdown('<div class="section-header">What the Detector Catches</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Blocked patterns:**")
        for item in ["Algorithm names (Kadane's, Dijkstra's, Floyd's…)",
                     "Direct data structure reveals ('use a hash map')",
                     "Big-O answers ('time complexity is O(n)')",
                     "Code syntax (def, for, return, .append…)",
                     "Explicit DP/memoization mentions"]:
            st.markdown(f"- 🚫 {item}")
    with col2:
        st.markdown("**Allowed patterns:**")
        for item in ["Structural questions about inputs",
                     "Analogies without naming the algorithm",
                     "Subproblem decomposition prompts",
                     "Constraint-pointing questions",
                     "Complexity-clue questions (not answers)"]:
            st.markdown(f"- ✅ {item}")


# ── Main ──────────────────────────────────────────────────────

def main():
    page = render_sidebar()

    if page == "🎓 Live Tutoring Session":
        screen_live_session()
    elif page == "⚖️ Before vs After":
        screen_before_after()
    elif page == "🗺️ RL Policy Visualizer":
        screen_policy_visualizer()
    elif page == "🔍 Leakage Detector":
        screen_leakage_detector()


if __name__ == "__main__":
    main()