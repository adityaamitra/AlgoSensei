"""
rl/ppo/ppo_agent.py
────────────────────
RL Module 2: PPO Curriculum Planning Agent
══════════════════════════════════════════

MATHEMATICAL FORMULATION
─────────────────────────
Clipped surrogate objective (Schulman et al., 2017):

    L^CLIP(θ) = E_t [ min(r_t(θ) Â_t, clip(r_t(θ), 1-ε, 1+ε) Â_t) ]

where:
    r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)   probability ratio
    Â_t    = GAE advantage estimate
    ε      = 0.2 (clip parameter)

Generalized Advantage Estimation (GAE, Schulman et al., 2016):
    Â_t = Σ_{k≥0} (γλ)^k δ_{t+k}
    δ_t = r_t + γ V(s_{t+1}) - V(s_t)

Full loss:
    L(θ) = L^CLIP - c_1 L^VF + c_2 H[π_θ]

State:  16-dim StudentKnowledgeState vector
Action: 42 discrete = 14 patterns × 3 difficulty deltas
"""

from __future__ import annotations
import os, yaml
from collections import deque
import numpy as np

_cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yaml")
with open(_cfg_path) as f:
    CFG = yaml.safe_load(f)

PATTERNS   = CFG["dsa_patterns"]
STATE_DIM  = CFG["ppo"]["state_dim"]
ACTION_DIM = CFG["ppo"]["action_dim"]  # 42
HIDDEN     = CFG["ppo"]["hidden"]

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

DELTAS = [-1, 0, 1]


def decode_action(action_id: int) -> tuple[str, int]:
    """Map flat action ID → (pattern_name, difficulty_delta)."""
    pattern = PATTERNS[action_id // 3]
    delta   = DELTAS[action_id % 3]
    return pattern, delta


# ── Actor-Critic Network ──────────────────────────────────────

if TORCH_OK:
    class ActorCritic(nn.Module):
        """
        Shared trunk → actor (policy) + critic (value) heads.
        Orthogonal weight init (standard PPO practice).
        Architecture: 16 → 128 → 64 → [42 logits | 1 value]
        """
        def __init__(self):
            super().__init__()
            h = HIDDEN
            self.trunk = nn.Sequential(
                nn.Linear(STATE_DIM, h[0]), nn.Tanh(),
                nn.Linear(h[0], h[1]),     nn.Tanh(),
            )
            self.actor  = nn.Linear(h[1], ACTION_DIM)
            self.critic = nn.Linear(h[1], 1)
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0.0)
            nn.init.orthogonal_(self.actor.weight, gain=0.01)

        def forward(self, x):
            f = self.trunk(x)
            return self.actor(f), self.critic(f).squeeze(-1)

        def act(self, state_t):
            logits, value = self(state_t)
            dist  = Categorical(logits=logits)
            action = dist.sample()
            return action, dist.log_prob(action), value, dist.entropy()

        def evaluate(self, states_t, actions_t):
            logits, values = self(states_t)
            dist = Categorical(logits=logits)
            return dist.log_prob(actions_t), values, dist.entropy()


# ── PPO Agent ─────────────────────────────────────────────────

class PPOCurriculumAgent:
    """
    PPO agent that learns an optimal tutoring curriculum.
    Trained on the StudentSimulator; deployed in TutoringOrchestrator.
    """

    def __init__(self, device: str = "cpu"):
        if not TORCH_OK:
            raise ImportError("PyTorch required for PPO agent. pip install torch")
        import torch
        self.device  = torch.device(device)
        self.net     = ActorCritic().to(self.device)
        self.opt     = optim.Adam([
            {"params": self.net.trunk.parameters(),  "lr": CFG["ppo"]["lr"]},
            {"params": self.net.actor.parameters(),  "lr": CFG["ppo"]["lr"]},
            {"params": self.net.critic.parameters(), "lr": CFG["ppo"]["lr"] * 0.5},
        ])
        # Rollout buffer
        self.buf_s, self.buf_a, self.buf_lp = [], [], []
        self.buf_r, self.buf_v, self.buf_d  = [], [], []
        # Logging
        self.episode_rewards: list[float] = []
        self.policy_losses:   list[float] = []
        self.value_losses:    list[float] = []

    def select_action(self, state: np.ndarray, exploit: bool = False
                      ) -> tuple[int, float, float]:
        import torch
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.net(s)
            if exploit:
                action = int(torch.argmax(logits).item())
                dist = Categorical(logits=logits)
                lp   = float(dist.log_prob(torch.tensor(action)).item())
            else:
                action, lp_t, v_t, _ = self.net.act(s)
                action = int(action.item())
                lp     = float(lp_t.item())
        return action, lp, float(value.item())

    def store(self, s, a, lp, r, v, done):
        self.buf_s.append(s); self.buf_a.append(a); self.buf_lp.append(lp)
        self.buf_r.append(r); self.buf_v.append(v); self.buf_d.append(done)

    def _compute_gae(self, next_v: float) -> tuple:
        import torch
        T   = len(self.buf_r)
        adv = np.zeros(T, np.float32)
        gae = 0.0
        γ   = CFG["ppo"]["gamma"]
        λ   = CFG["ppo"]["gae_lambda"]
        for t in reversed(range(T)):
            nv  = next_v if t == T-1 else self.buf_v[t+1]
            nd  = float(self.buf_d[t+1]) if t < T-1 else 0.0
            δ   = self.buf_r[t] + γ * nv * (1-nd) - self.buf_v[t]
            gae = δ + γ * λ * (1-nd) * gae
            adv[t] = gae
        ret = adv + np.array(self.buf_v, dtype=np.float32)
        return torch.FloatTensor(adv), torch.FloatTensor(ret)

    def update(self, next_v: float = 0.0) -> dict:
        import torch
        # Guard: skip update on tiny buffers — causes std()=NaN (PyTorch warning)
        if len(self.buf_r) < 2:
            self.buf_s.clear(); self.buf_a.clear(); self.buf_lp.clear()
            self.buf_r.clear(); self.buf_v.clear(); self.buf_d.clear()
            return {"pl": 0.0, "vl": 0.0}

        adv, ret = self._compute_gae(next_v)

        # Guard: skip if GAE produced NaNs (reward explosion)
        if torch.isnan(adv).any() or torch.isinf(adv).any():
            self.buf_s.clear(); self.buf_a.clear(); self.buf_lp.clear()
            self.buf_r.clear(); self.buf_v.clear(); self.buf_d.clear()
            return {"pl": 0.0, "vl": 0.0}

        # Safe normalization: only divide by std if it is meaningful
        adv_std = adv.std()
        if adv_std > 1e-8:
            adv = (adv - adv.mean()) / (adv_std + 1e-8)
        else:
            adv = adv - adv.mean()

        states_t = torch.FloatTensor(np.array(self.buf_s)).to(self.device)
        acts_t   = torch.LongTensor(self.buf_a).to(self.device)
        old_lp_t = torch.FloatTensor(self.buf_lp).to(self.device)
        adv      = adv.to(self.device)
        ret      = ret.to(self.device)

        eps = CFG["ppo"]["clip_eps"]
        c1  = CFG["ppo"]["value_coef"]
        c2  = CFG["ppo"]["entropy_coef"]
        T   = len(self.buf_r)
        bs  = CFG["ppo"]["batch_size"]
        metrics = {"pl": 0.0, "vl": 0.0}

        for _ in range(CFG["ppo"]["update_epochs"]):
            idx = np.random.permutation(T)
            for start in range(0, T, bs):
                b = idx[start:start+bs]
                new_lp, vals, ent = self.net.evaluate(states_t[b], acts_t[b])
                ratio = torch.exp(new_lp - old_lp_t[b])
                s1 = ratio * adv[b]
                s2 = torch.clamp(ratio, 1-eps, 1+eps) * adv[b]
                pl = -torch.min(s1, s2).mean()
                vl = nn.functional.mse_loss(vals, ret[b])
                loss = pl + c1*vl - c2*ent.mean()
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(),
                                         CFG["ppo"]["grad_norm"])
                self.opt.step()
                metrics["pl"] += pl.item()
                metrics["vl"] += vl.item()

        self.buf_s.clear(); self.buf_a.clear(); self.buf_lp.clear()
        self.buf_r.clear(); self.buf_v.clear(); self.buf_d.clear()
        return metrics

    def save(self, path: str):
        import torch
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "net": self.net.state_dict(),
            "opt": self.opt.state_dict(),
            "episode_rewards": self.episode_rewards,
            "policy_losses":   self.policy_losses,
            "value_losses":    self.value_losses,
        }, path)

    def load(self, path: str):
        import torch
        ck = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ck["net"])
        self.opt.load_state_dict(ck["opt"])
        self.episode_rewards = ck.get("episode_rewards", [])
        self.policy_losses   = ck.get("policy_losses", [])
        self.value_losses    = ck.get("value_losses", [])


# ── Training loop ─────────────────────────────────────────────

def train_ppo(n_episodes: int = None, seed: int = 42,
              verbose: bool = True) -> PPOCurriculumAgent:
    import torch
    from environment.tutoring_env import SessionEnvironment

    n_ep     = n_episodes or CFG["ppo"]["n_episodes"]
    max_s    = CFG["ppo"]["max_steps"]
    buf_size = CFG["ppo"]["buffer_size"]

    torch.manual_seed(seed)
    env   = SessionEnvironment(seed=seed)
    agent = PPOCurriculumAgent()
    rw100 = deque(maxlen=100)

    for ep in range(n_ep):
        student   = env.new_student()
        ep_reward = 0.0
        prev_hint = 3.0

        for step in range(max_s):
            sv     = student.to_vector()
            a, lp, v = agent.select_action(sv)
            pattern, delta = decode_action(a)

            reward, result = env.run_session(
                student, pattern, delta, "socratic", prev_hint
            )
            prev_hint = result.avg_hints_per_problem
            done = (step == max_s - 1) or (not result.student_retained)

            # Clip rewards to prevent exploding advantages → NaN in GAE
            reward = float(np.clip(reward, -20.0, 20.0))
            agent.store(sv, a, lp, reward, v, done)
            ep_reward += reward

            if (len(agent.buf_r) >= buf_size or done) and len(agent.buf_r) >= 2:
                nv = 0.0
                if not done:
                    nv = agent.select_action(student.to_vector())[2]
                m = agent.update(nv)
                agent.policy_losses.append(m["pl"])
                agent.value_losses.append(m["vl"])

            if done:
                break

        agent.episode_rewards.append(ep_reward)
        rw100.append(ep_reward)

        if verbose and (ep + 1) % 500 == 0:
            print(f"  Episode {ep+1:5d} | Mean reward (100): {np.mean(rw100):7.3f}")

    return agent