"""
modules/module5_marl.py
Module 5 — QMIX Multi-Agent Reinforcement Learning for Water Allocation

Three competing agents share a limited water budget:
  Agent 0: Agriculture   (largest consumer, ~80% of India's water use)
  Agent 1: Industry      (~10% of water use)
  Agent 2: Municipal     (~10%, but highest social priority)

Each agent observes local water conditions and chooses an allocation
request. QMIX (Monotonic Value Function Factorisation) trains them
cooperatively — maximising a shared reward that balances:
  - Efficiency:     meet economic/social demands
  - Equity:         no agent is severely deprived
  - Sustainability: groundwater levels remain above critical threshold

Paper: Rashid et al., "QMIX: Monotonic Value Function Factorisation
       for Deep Multi-Agent Reinforcement Learning", ICML 2018
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
from loguru import logger
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import get_device, save_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Water Allocation Environment
# ─────────────────────────────────────────────────────────────────────────────

class WaterAllocationEnv:
    """
    Multi-agent water allocation environment for the Ganges basin.

    State (per agent): [cwsi, groundwater, precipitation, demand, reservoir_level,
                        season_sin, season_cos, agent_id_onehot(3)]  → 11 dims
    Global state:      concatenation of all agent states                → 33 dims

    Action: discrete allocation level 0..4
        0: Very low  (20% of max demand)
        1: Low       (40%)
        2: Medium    (60%)
        3: High      (80%)
        4: Maximum   (100%)

    Reward: shared (cooperative) + local (individual)
        r = w_eff * efficiency - w_eq * equity_penalty - w_sus * sustainability_penalty
    """

    NUM_AGENTS     = 3
    ACTION_DIM     = 5
    LOCAL_OBS_DIM  = 11
    GLOBAL_STATE_DIM = LOCAL_OBS_DIM * NUM_AGENTS

    AGENT_NAMES  = ["Agriculture", "Industry", "Municipal"]
    MAX_DEMANDS  = [8.0, 1.5, 1.0]    # km³/month per agent
    PRIORITIES   = [0.50, 0.20, 0.30]  # social priority weights
    ALLOC_LEVELS = [0.20, 0.40, 0.60, 0.80, 1.00]

    def __init__(self, dataset_dict: dict, cfg: dict):
        self.data    = dataset_dict["data"]   # (T, N, F)
        self.cwsi    = dataset_dict["cwsi"]   # (T, N)
        self.cfg     = cfg["marl"]
        self.T, self.N, self.F = self.data.shape
        self.t       = 0
        self.water_budget = None
        self.agent_states = None
        self._rng = np.random.RandomState(42)

    def _get_basin_state(self):
        """Aggregate current basin conditions (mean over nodes)."""
        d = self.data[self.t].mean(axis=0)  # (F,)
        cwsi_mean = self.cwsi[self.t].mean()
        return d, cwsi_mean

    def _season_encoding(self):
        """Encode month as sin/cos (cyclical feature)."""
        month = (self.t % 12) + 1
        return np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)

    def reset(self, t_start: int = None):
        """Reset to a random or specified timestep."""
        if t_start is None:
            self.t = self._rng.randint(12, self.T - 24)
        else:
            self.t = t_start
        self.water_budget = self._compute_water_budget()
        return self._get_observations()

    def _compute_water_budget(self) -> float:
        """
        Available water = function of precipitation + groundwater + reservoir.
        Higher CWSI → lower available budget.
        """
        d, cwsi = self._get_basin_state()
        # F indices: 0=gw, 1=prec, 5=discharge, 6=reservoir
        base_budget = (d[1] * 2.5 + max(d[0] + 10, 0) * 0.5 +
                       d[6] * 0.05 * 10)
        # Crisis penalty
        stress_factor = 1 - 0.6 * cwsi
        budget = max(base_budget * stress_factor, 1.0)
        return float(budget)

    def _get_observations(self) -> list:
        """Return local observation for each agent."""
        d, cwsi = self._get_basin_state()
        sin_s, cos_s = self._season_encoding()
        obs_list = []
        for a in range(self.NUM_AGENTS):
            agent_onehot = np.zeros(self.NUM_AGENTS)
            agent_onehot[a] = 1.0
            obs = np.array([
                cwsi,               # water stress
                d[0] / 20,          # normalised groundwater anomaly
                d[1] / 15,          # normalised precipitation
                self.MAX_DEMANDS[a] / 10,  # agent's max demand
                d[6] / 100,         # reservoir level
                self.water_budget / 20,    # available budget
                sin_s, cos_s,       # season
                *agent_onehot,      # agent identity
            ], dtype=np.float32)
            obs_list.append(obs)
        return obs_list

    def _get_global_state(self) -> np.ndarray:
        """Global state = concatenation of all local observations."""
        return np.concatenate(self._get_observations()).astype(np.float32)

    def step(self, actions: list) -> tuple:
        """
        actions: list of ints [a0, a1, a2], one per agent
        Returns: (next_obs, global_state, rewards, done, info)
        """
        alloc_fracs  = [self.ALLOC_LEVELS[a] for a in actions]
        allocations  = [f * d for f, d in zip(alloc_fracs, self.MAX_DEMANDS)]
        total_request = sum(allocations)

        # Clip to budget (proportional rationing if over-requested)
        if total_request > self.water_budget:
            ratio = self.water_budget / (total_request + 1e-8)
            allocations = [a * ratio for a in allocations]

        # ─ Compute rewards ─
        w = self.cfg["reward_weights"]

        # Efficiency: fraction of economic demand met
        efficiency = sum(
            self.PRIORITIES[a] * (allocations[a] / (self.MAX_DEMANDS[a] + 1e-8))
            for a in range(self.NUM_AGENTS)
        )

        # Equity: penalise large disparities (Gini-like)
        fracs = np.array([allocations[a] / (self.MAX_DEMANDS[a] + 1e-8)
                          for a in range(self.NUM_AGENTS)])
        equity_penalty = np.std(fracs)

        # Sustainability: penalise if budget is overdrawn or GW is very negative
        d, cwsi = self._get_basin_state()
        sus_penalty = max(0, cwsi - 0.70) + max(0, -d[0] / 10)

        # Shared global reward
        global_reward = (w["efficiency"]    *  efficiency
                       - w["equity"]        *  equity_penalty
                       - w["sustainability"]*  sus_penalty)

        # Individual rewards (local efficiency per agent)
        local_rewards = [
            w["efficiency"] * self.PRIORITIES[a] * fracs[a]
            for a in range(self.NUM_AGENTS)
        ]
        # Total reward = global + local blend
        rewards = [0.7 * global_reward + 0.3 * lr for lr in local_rewards]

        # Advance time
        self.t = min(self.t + 1, self.T - 2)
        self.water_budget = self._compute_water_budget()
        done = (self.t >= self.T - 2)

        next_obs      = self._get_observations()
        global_state  = self._get_global_state()

        info = {
            "allocations":     allocations,
            "efficiency":      efficiency,
            "equity_penalty":  equity_penalty,
            "sus_penalty":     sus_penalty,
            "water_budget":    self.water_budget,
            "cwsi":            cwsi,
        }
        return next_obs, global_state, rewards, done, info


# ─────────────────────────────────────────────────────────────────────────────
# Agent Networks
# ─────────────────────────────────────────────────────────────────────────────

class AgentNetwork(nn.Module):
    """
    Individual agent Q-network (DRQN-style with GRU for partial observability).
    Estimates Q(observation, action) for each agent independently.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru     = nn.GRUCell(hidden_dim, hidden_dim)
        self.q_head  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim)

    def forward(self, obs: torch.Tensor, h: torch.Tensor) -> tuple:
        """
        obs: (B, obs_dim)
        h:   (B, hidden_dim)
        Returns: q_values (B, action_dim), h_new (B, hidden_dim)
        """
        x   = self.encoder(obs)
        h   = self.gru(x, h)
        q   = self.q_head(h)
        return q, h


# ─────────────────────────────────────────────────────────────────────────────
# QMIX Mixing Network
# ─────────────────────────────────────────────────────────────────────────────

class QMIXMixer(nn.Module):
    """
    QMIX mixing network: combines individual Q-values into a joint Q-value.
    Key constraint: dQ_tot/dQ_i ≥ 0 (monotonic) — enforced by ELU + abs weights.

    Hypernetwork conditions mixing weights on the global state.
    """
    def __init__(self, num_agents: int, state_dim: int, embed_dim: int = 32):
        super().__init__()
        self.num_agents = num_agents
        self.embed_dim  = embed_dim

        # Hypernetworks produce weights for two linear layers
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_agents * embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        agent_qs: (B, T, num_agents)
        state:    (B, T, state_dim)
        Returns:  Q_tot (B, T, 1)
        """
        B, T, n = agent_qs.shape
        agent_qs = agent_qs.reshape(B * T, 1, n)
        state    = state.reshape(B * T, -1)

        # Layer 1 (weights are absolute for monotonicity)
        w1 = torch.abs(self.hyper_w1(state)).reshape(B * T, n, self.embed_dim)
        b1 = self.hyper_b1(state).unsqueeze(1)
        h  = F.elu(torch.bmm(agent_qs, w1) + b1)   # (BT, 1, embed)

        # Layer 2
        w2 = torch.abs(self.hyper_w2(state)).reshape(B * T, self.embed_dim, 1)
        b2 = self.hyper_b2(state).reshape(B * T, 1, 1)
        q_tot = torch.bmm(h, w2) + b2               # (BT, 1, 1)

        return q_tot.reshape(B, T, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Experience Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

Transition = namedtuple("Transition",
    ["obs", "global_state", "actions", "rewards", "next_obs", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# QMIX Trainer
# ─────────────────────────────────────────────────────────────────────────────

class QMIXTrainer:
    """Full QMIX training loop with epsilon-greedy exploration."""

    def __init__(self, env: WaterAllocationEnv, cfg: dict):
        self.env     = env
        self.cfg     = cfg["marl"]
        self.device  = get_device()
        n_agents     = env.NUM_AGENTS
        obs_dim      = env.LOCAL_OBS_DIM
        action_dim   = env.ACTION_DIM
        state_dim    = env.GLOBAL_STATE_DIM
        embed_dim    = self.cfg["mixing_embed_dim"]

        # Agent networks (one per agent) + target networks
        self.agents  = nn.ModuleList([
            AgentNetwork(obs_dim, action_dim).to(self.device)
            for _ in range(n_agents)
        ])
        self.target_agents = nn.ModuleList([
            AgentNetwork(obs_dim, action_dim).to(self.device)
            for _ in range(n_agents)
        ])

        # Mixing networks
        self.mixer        = QMIXMixer(n_agents, state_dim, embed_dim).to(self.device)
        self.target_mixer = QMIXMixer(n_agents, state_dim, embed_dim).to(self.device)

        # Sync targets
        self._update_targets(tau=1.0)

        # Optimiser covers all agent nets + mixer
        all_params = list(self.agents.parameters()) + list(self.mixer.parameters())
        self.optimiser = torch.optim.AdamW(all_params, lr=self.cfg["lr"])

        self.buffer  = ReplayBuffer(self.cfg["buffer_size"])
        self.gamma   = self.cfg["gamma"]
        self.epsilon = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.9995

    def _update_targets(self, tau: float = 0.005) -> None:
        """Soft target update: θ' ← τθ + (1-τ)θ'"""
        for net, target in [(self.agents, self.target_agents),
                             (self.mixer, self.target_mixer)]:
            if isinstance(net, nn.ModuleList):
                for ag, tg in zip(net, target):
                    for p, tp in zip(ag.parameters(), tg.parameters()):
                        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
            else:
                for p, tp in zip(net.parameters(), target.parameters()):
                    tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    def select_actions(self, obs_list: list, h_list: list) -> tuple:
        """Epsilon-greedy action selection."""
        actions  = []
        new_h    = []
        for a in range(self.env.NUM_AGENTS):
            obs_t = torch.FloatTensor(obs_list[a]).unsqueeze(0).to(self.device)
            h_t   = h_list[a]
            q, h_new = self.agents[a](obs_t, h_t)
            new_h.append(h_new)
            if np.random.random() < self.epsilon:
                actions.append(np.random.randint(self.env.ACTION_DIM))
            else:
                actions.append(q.argmax(dim=-1).item())
        return actions, new_h

    def train_step(self, batch_size: int = 64) -> float:
        if len(self.buffer) < batch_size:
            return 0.0

        batch = self.buffer.sample(batch_size)

        obs_batch      = torch.FloatTensor(np.array(batch.obs)).to(self.device)
        state_batch    = torch.FloatTensor(np.array(batch.global_state)).to(self.device)
        actions_batch  = torch.LongTensor(np.array(batch.actions)).to(self.device)
        rewards_batch  = torch.FloatTensor(np.array(batch.rewards)).to(self.device)
        next_obs_batch = torch.FloatTensor(np.array(batch.next_obs)).to(self.device)
        next_state_b   = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch     = torch.FloatTensor(np.array(batch.done)).to(self.device)

        # obs shape: (B, num_agents, obs_dim) → add time dim T=1 for QMIX
        B  = batch_size
        n  = self.env.NUM_AGENTS

        # Current Q-values
        agent_qs = []
        for a in range(n):
            h = self.agents[a].init_hidden(B).to(self.device)
            q, _ = self.agents[a](obs_batch[:, a], h)
            q_a  = q.gather(1, actions_batch[:, a].unsqueeze(-1)).squeeze(-1)
            agent_qs.append(q_a)
        agent_qs = torch.stack(agent_qs, dim=-1).unsqueeze(1)  # (B, 1, n)
        q_tot    = self.mixer(agent_qs, state_batch.unsqueeze(1)).squeeze()  # (B,)

        # Target Q-values
        with torch.no_grad():
            tgt_qs = []
            for a in range(n):
                h = self.target_agents[a].init_hidden(B).to(self.device)
                q_tgt, _ = self.target_agents[a](next_obs_batch[:, a], h)
                tgt_qs.append(q_tgt.max(dim=-1).values)
            tgt_qs   = torch.stack(tgt_qs, dim=-1).unsqueeze(1)  # (B, 1, n)
            q_tot_tgt = self.target_mixer(tgt_qs, next_state_b.unsqueeze(1)).squeeze()

            # Mean reward across agents
            mean_reward = rewards_batch.mean(dim=-1)
            done_f = done_batch[:, 0] if done_batch.dim() > 1 else done_batch
            y = mean_reward + self.gamma * q_tot_tgt * (1 - done_f)

        loss = F.mse_loss(q_tot, y)
        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.agents.parameters()) + list(self.mixer.parameters()), 10.0)
        self.optimiser.step()
        self._update_targets(tau=0.005)
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        return loss.item()

    def train(self, num_steps: int = 100000) -> dict:
        """Full QMIX training loop."""
        logger.info("── Training Module 5: QMIX MARL ───────────────────")
        episode_rewards = []
        losses = []
        step = 0

        while step < num_steps:
            obs = self.env.reset()
            h_list = [self.agents[a].init_hidden(1).to(self.device)
                      for a in range(self.env.NUM_AGENTS)]
            ep_reward = 0.0
            done = False

            while not done and step < num_steps:
                global_state = self.env._get_global_state()
                actions, h_list = self.select_actions(obs, h_list)

                next_obs, next_state, rewards, done, info = self.env.step(actions)

                # Store transition
                self.buffer.push(
                    np.array(obs),           # (n_agents, obs_dim)
                    global_state,
                    np.array(actions),
                    np.array(rewards),
                    np.array(next_obs),
                    next_state,
                    float(done),
                )

                obs = next_obs
                ep_reward += np.mean(rewards)
                step += 1

                if step % 4 == 0:
                    loss = self.train_step(self.cfg["batch_size"])
                    losses.append(loss)

            episode_rewards.append(ep_reward)

            if len(episode_rewards) % 50 == 0:
                mean_r = np.mean(episode_rewards[-50:])
                logger.info(f"Step {step:6d}/{num_steps} | "
                            f"Episode reward (50-ep mean): {mean_r:.3f} | "
                            f"Epsilon: {self.epsilon:.3f}")

        # Save
        torch.save({
            "agents":  [a.state_dict() for a in self.agents],
            "mixer":   self.mixer.state_dict(),
        }, "models/checkpoints/qmix_final.pt")
        logger.success(f"Module 5 complete. Final mean reward: {np.mean(episode_rewards[-100:]):.3f}")

        return {"episode_rewards": episode_rewards, "losses": losses}

    def evaluate(self, n_episodes: int = 10) -> dict:
        """
        Evaluate the trained policy (greedy, no exploration).
        Returns allocation statistics across episodes.
        """
        self.epsilon = 0.0   # greedy
        all_infos = []
        for ep in range(n_episodes):
            obs = self.env.reset()
            h_list = [self.agents[a].init_hidden(1).to(self.device)
                      for a in range(self.env.NUM_AGENTS)]
            done = False
            ep_infos = []
            while not done:
                actions, h_list = self.select_actions(obs, h_list)
                obs, _, _, done, info = self.env.step(actions)
                ep_infos.append(info)
            all_infos.extend(ep_infos)

        # Summarise
        efficiencies   = [i["efficiency"]     for i in all_infos]
        equity_pen     = [i["equity_penalty"] for i in all_infos]
        sus_pen        = [i["sus_penalty"]    for i in all_infos]
        logger.info(f"Eval: efficiency={np.mean(efficiencies):.3f}, "
                    f"equity_penalty={np.mean(equity_pen):.3f}, "
                    f"sus_penalty={np.mean(sus_pen):.3f}")
        return {
            "mean_efficiency":    np.mean(efficiencies),
            "mean_equity_pen":    np.mean(equity_pen),
            "mean_sus_pen":       np.mean(sus_pen),
            "all_infos":          all_infos,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Module 5 Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_marl_module(dataset_dict: dict, cfg: dict) -> dict:
    """Entry point for Module 5."""
    logger.info("── Initialising Module 5: MARL ─────────────────────")
    env     = WaterAllocationEnv(dataset_dict, cfg)
    trainer = QMIXTrainer(env, cfg)
    results = trainer.train(num_steps=cfg["marl"]["train_steps"])
    eval_results = trainer.evaluate(n_episodes=20)
    return {
        "trainer":      trainer,
        "train_results": results,
        "eval_results": eval_results,
    }
