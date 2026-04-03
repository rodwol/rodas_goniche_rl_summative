"""
pg_training.py
==============
Trains three policy-gradient RL agents on PharmacySupplyEnv-v0:
  1. REINFORCE  — classic Monte Carlo policy gradient (custom PyTorch)
  2. PPO        — Proximal Policy Optimisation (Stable Baselines 3)
  3. A2C        — Advantage Actor-Critic (Stable Baselines 3)

What is Policy Gradient?
------------------------
Unlike DQN which learns the VALUE of actions, policy gradient methods
directly learn a POLICY — a function that maps states to action probabilities.

  DQN  : "action A scores 80, action B scores 30 → pick A"
  PG   : "in state X, take action A with 70% probability, action B with 30%"

The three algorithms differ in HOW they update the policy:

  REINFORCE : Wait until the episode ends, then update based on total return.
              Simple but high variance — like grading a student only at the end
              of a year-long exam.

  PPO       : Update during the episode, but CLIP the update size to prevent
              catastrophic policy changes. The "clipping" is the key innovation —
              it keeps training stable.
              epsilon (clip_range) controls how much the policy can change per step.

  A2C       : Uses an ACTOR (policy) + CRITIC (value estimator) simultaneously.
              The critic tells the actor "that action was better/worse than average"
              reducing the variance of updates compared to REINFORCE.

Usage:
    python training/pg_training.py
    python training/pg_training.py --algo ppo
    python training/pg_training.py --algo a2c
    python training/pg_training.py --algo reinforce
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path setup ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

from environment.custom_env import PharmacySupplyEnv

# ── Output directories ────────────────────────────────────────────────────
for d in ["models/pg/reinforce", "models/pg/ppo", "models/pg/a2c",
          "models/pg/plots"]:
    os.makedirs(d, exist_ok=True)

PLOT_DIR = "models/pg/plots"


# ═════════════════════════════════════════════════════════════════════════
# SECTION 1 — REINFORCE (custom PyTorch implementation)
# ═════════════════════════════════════════════════════════════════════════

class PolicyNetwork(nn.Module):
    """
    A simple feedforward neural network that outputs action probabilities.

    Architecture:
        Input (7 obs) → Hidden layer → Hidden layer → Output (5 actions)

    The softmax at the end converts raw scores into probabilities that
    sum to 1.0 — so the agent can sample actions proportionally.
    """
    def __init__(self, obs_dim=7, action_dim=5, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)      # output = action probabilities
        )

    def forward(self, x):
        return self.network(x)


def compute_returns(rewards, gamma):
    """
    Compute discounted returns for a full episode (Monte Carlo return).

    For each time step t, the return G_t is:
        G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

    We normalise the returns (subtract mean, divide by std) to reduce
    variance and stabilise training — this is standard practice.

    Args:
        rewards : list of rewards from one complete episode
        gamma   : discount factor (0 = only care about now, 1 = full future)

    Returns:
        returns : normalised discounted returns as a torch tensor
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):        # work backwards from episode end
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)

    # Normalise to reduce variance
    if returns.std() > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns


def train_reinforce(run_id, hyperparams, n_episodes=400, seed=42):
    """
    Train a REINFORCE agent for a fixed number of episodes.

    REINFORCE algorithm:
      For each episode:
        1. Run the full episode using current policy
        2. Compute discounted return G_t for each step
        3. Update policy: push up probability of actions that led to
           high returns, push down probability of low-return actions
        4. Entropy bonus: encourage exploration by rewarding uncertainty

    Args:
        run_id      : integer label (1-10) for this hyperparameter run
        hyperparams : dict with learning_rate, gamma, hidden_size,
                      entropy_coeff
        n_episodes  : number of episodes to train
        seed        : random seed

    Returns:
        episode_rewards : list of total reward per episode
        mean_reward     : average over last 20 episodes
        std_reward      : std over last 20 episodes
        entropy_history : list of mean entropy per episode
    """
    print(f"\n{'='*60}")
    print(f"  REINFORCE Run {run_id:02d} | "
          f"LR={hyperparams['learning_rate']} | "
          f"gamma={hyperparams['gamma']} | "
          f"hidden={hyperparams['hidden_size']} | "
          f"entropy_coeff={hyperparams['entropy_coeff']}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = PharmacySupplyEnv(seed=seed)
    policy = PolicyNetwork(
        obs_dim=7,
        action_dim=5,
        hidden_size=hyperparams["hidden_size"]
    )
    optimiser = optim.Adam(policy.parameters(),
                           lr=hyperparams["learning_rate"])

    episode_rewards = []
    entropy_history = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        log_probs   = []    # log probability of chosen action at each step
        rewards     = []    # reward at each step
        entropies   = []    # entropy of policy distribution at each step

        # ── Roll out one full episode ─────────────────────────────────
        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action_probs = policy(obs_tensor)           # shape: (1, 5)
            dist = Categorical(action_probs)            # discrete distribution
            action = dist.sample()                      # sample one action

            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())

            obs, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated

        # ── Compute discounted returns ────────────────────────────────
        returns = compute_returns(rewards, hyperparams["gamma"])

        # ── Policy gradient loss ──────────────────────────────────────
        # Loss = -sum(log_prob * return) - entropy_coeff * entropy
        # The negative sign: we MINIMISE loss, but we want to MAXIMISE reward
        # Entropy bonus: encourages the agent to stay exploratory (not greedy)
        log_probs_tensor = torch.stack(log_probs)
        entropies_tensor = torch.stack(entropies)

        pg_loss      = -(log_probs_tensor * returns).mean()
        entropy_loss = -hyperparams["entropy_coeff"] * entropies_tensor.mean()
        loss = pg_loss + entropy_loss

        # ── Backpropagation ───────────────────────────────────────────
        optimiser.zero_grad()
        loss.backward()
        # Gradient clipping: prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimiser.step()

        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        entropy_history.append(entropies_tensor.mean().item())

        # Print progress every 50 episodes
        if (episode + 1) % 50 == 0:
            recent = episode_rewards[-20:]
            print(f"  Episode {episode+1:4d}/{n_episodes} | "
                  f"Mean reward (last 20): {np.mean(recent):7.1f} | "
                  f"Entropy: {entropy_history[-1]:.3f}")

    env.close()

    last_20     = episode_rewards[-20:]
    mean_reward = np.mean(last_20)
    std_reward  = np.std(last_20)

    # Save policy weights
    save_path = f"models/pg/reinforce/reinforce_run_{run_id:02d}.pt"
    torch.save(policy.state_dict(), save_path)
    print(f"\n  Run {run_id:02d} complete | "
          f"Mean reward: {mean_reward:.1f} ± {std_reward:.1f}")
    print(f"  Model saved to: {save_path}")

    return episode_rewards, mean_reward, std_reward, entropy_history


# ═════════════════════════════════════════════════════════════════════════
# SECTION 2 — PPO and A2C via Stable Baselines 3
# ═════════════════════════════════════════════════════════════════════════

class RewardLoggerCallback(BaseCallback):
    """Records episode rewards during SB3 training."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards  = []
        self.entropy_history  = []
        self._current_rewards = []

    def _on_step(self) -> bool:
        self._current_rewards.append(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self.episode_rewards.append(sum(self._current_rewards))
            self._current_rewards = []
        return True


def train_sb3_algo(algo_name, run_id, hyperparams,
                   total_timesteps=80_000, seed=42):
    """
    Train a PPO or A2C agent using Stable Baselines 3.

    Args:
        algo_name       : "ppo" or "a2c"
        run_id          : integer 1-10
        hyperparams     : dict of algorithm hyperparameters
        total_timesteps : training budget
        seed            : random seed

    Returns:
        model, callback, mean_reward, std_reward
    """
    algo_name = algo_name.lower()
    assert algo_name in ("ppo", "a2c"), "algo_name must be 'ppo' or 'a2c'"

    AlgoClass = PPO if algo_name == "ppo" else A2C
    save_dir  = f"models/pg/{algo_name}"

    print(f"\n{'='*60}")
    print(f"  {algo_name.upper()} Run {run_id:02d} | "
          + " | ".join(f"{k}={v}" for k, v in hyperparams.items()))
    print(f"{'='*60}")

    env      = Monitor(PharmacySupplyEnv(seed=seed))
    eval_env = Monitor(PharmacySupplyEnv(seed=seed + 100))

    reward_callback = RewardLoggerCallback()
    eval_callback   = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir,
                                          f"run_{run_id:02d}_best"),
        log_path=os.path.join(save_dir, f"run_{run_id:02d}_logs"),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=0,
    )

    # ── Build model ───────────────────────────────────────────────────────
    if algo_name == "ppo":
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=hyperparams["learning_rate"],
            gamma=hyperparams["gamma"],
            n_steps=hyperparams["n_steps"],
            batch_size=hyperparams["batch_size"],
            n_epochs=hyperparams["n_epochs"],
            clip_range=hyperparams["clip_range"],
            ent_coef=hyperparams["ent_coef"],
            vf_coef=hyperparams["vf_coef"],
            gae_lambda=hyperparams["gae_lambda"],
            verbose=1,
            seed=seed,
            device="auto",
        )
    else:  # A2C
        model = A2C(
            policy="MlpPolicy",
            env=env,
            learning_rate=hyperparams["learning_rate"],
            gamma=hyperparams["gamma"],
            n_steps=hyperparams["n_steps"],
            ent_coef=hyperparams["ent_coef"],
            vf_coef=hyperparams["vf_coef"],
            gae_lambda=hyperparams["gae_lambda"],
            max_grad_norm=hyperparams["max_grad_norm"],
            verbose=1,
            seed=seed,
            device="auto",
        )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[reward_callback, eval_callback],
        progress_bar=True,
    )

    last_20     = reward_callback.episode_rewards[-20:]
    mean_reward = np.mean(last_20) if last_20 else 0.0
    std_reward  = np.std(last_20)  if last_20 else 0.0

    save_path = os.path.join(save_dir, f"{algo_name}_run_{run_id:02d}")
    model.save(save_path)
    print(f"\n  Run {run_id:02d} complete | "
          f"Mean reward: {mean_reward:.1f} ± {std_reward:.1f}")
    print(f"  Model saved to: {save_path}.zip")

    env.close()
    eval_env.close()

    return model, reward_callback, mean_reward, std_reward


# ═════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER TABLES
# ═════════════════════════════════════════════════════════════════════════

REINFORCE_RUNS = [
    # Run 1 — Baseline
    {"learning_rate": 1e-3, "gamma": 0.99,
     "hidden_size": 64,  "entropy_coeff": 0.01},
    # Run 2 — Lower LR (slower, more stable)
    {"learning_rate": 1e-4, "gamma": 0.99,
     "hidden_size": 64,  "entropy_coeff": 0.01},
    # Run 3 — Higher LR (aggressive updates)
    {"learning_rate": 5e-3, "gamma": 0.99,
     "hidden_size": 64,  "entropy_coeff": 0.01},
    # Run 4 — Low gamma (short-sighted)
    {"learning_rate": 1e-3, "gamma": 0.85,
     "hidden_size": 64,  "entropy_coeff": 0.01},
    # Run 5 — Very low gamma
    {"learning_rate": 1e-3, "gamma": 0.70,
     "hidden_size": 64,  "entropy_coeff": 0.01},
    # Run 6 — Larger network
    {"learning_rate": 1e-3, "gamma": 0.99,
     "hidden_size": 128, "entropy_coeff": 0.01},
    # Run 7 — Smaller network
    {"learning_rate": 1e-3, "gamma": 0.99,
     "hidden_size": 32,  "entropy_coeff": 0.01},
    # Run 8 — High entropy bonus (more exploration)
    {"learning_rate": 1e-3, "gamma": 0.99,
     "hidden_size": 64,  "entropy_coeff": 0.05},
    # Run 9 — No entropy bonus (greedy)
    {"learning_rate": 1e-3, "gamma": 0.99,
     "hidden_size": 64,  "entropy_coeff": 0.0},
    # Run 10 — Best combination
    {"learning_rate": 1e-4, "gamma": 0.99,
     "hidden_size": 128, "entropy_coeff": 0.02},
]

PPO_RUNS = [
    # Run 1 — Baseline
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 2048,
     "batch_size": 64, "n_epochs": 10, "clip_range": 0.2,
     "ent_coef": 0.01, "vf_coef": 0.5, "gae_lambda": 0.95},
    # Run 2 — Lower LR
    {"learning_rate": 1e-4, "gamma": 0.99, "n_steps": 2048,
     "batch_size": 64, "n_epochs": 10, "clip_range": 0.2,
     "ent_coef": 0.01, "vf_coef": 0.5, "gae_lambda": 0.95},
    # Run 3 — Higher LR
    {"learning_rate": 1e-3, "gamma": 0.99, "n_steps": 2048,
     "batch_size": 64, "n_epochs": 10, "clip_range": 0.2,
     "ent_coef": 0.01, "vf_coef": 0.5, "gae_lambda": 0.95},
    # Run 4 — Narrow clip range (conservative updates)
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 2048,
     "batch_size": 64, "n_epochs": 10, "clip_range": 0.1,
     "ent_coef": 0.01, "vf_coef": 0.5, "gae_lambda": 0.95},
    # Run 5 — Wide clip range (aggressive updates)
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 2048,
     "batch_size": 64, "n_epochs": 10, "clip_range": 0.3,
     "ent_coef": 0.01, "vf_coef": 0.5, "gae_lambda": 0.95},
    # Run 6 — More PPO epochs per update
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 2048,
     "batch_size": 64, "n_epochs": 20, "clip_range": 0.2,
     "ent_coef": 0.01, "vf_coef": 0.5, "gae_lambda": 0.95},
    # Run 7 — Shorter rollout (updates more frequently)
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 512,
     "batch_size": 64, "n_epochs": 10, "clip_range": 0.2,
     "ent_coef": 0.01, "vf_coef": 0.5, "gae_lambda": 0.95},
    # Run 8 — High entropy (explore more)
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 2048,
     "batch_size": 64, "n_epochs": 10, "clip_range": 0.2,
     "ent_coef": 0.05, "vf_coef": 0.5, "gae_lambda": 0.95},
    # Run 9 — Low gamma
    {"learning_rate": 3e-4, "gamma": 0.90, "n_steps": 2048,
     "batch_size": 64, "n_epochs": 10, "clip_range": 0.2,
     "ent_coef": 0.01, "vf_coef": 0.5, "gae_lambda": 0.95},
    # Run 10 — Best combination
    {"learning_rate": 1e-4, "gamma": 0.99, "n_steps": 2048,
     "batch_size": 128, "n_epochs": 15, "clip_range": 0.15,
     "ent_coef": 0.02, "vf_coef": 0.5, "gae_lambda": 0.98},
]

A2C_RUNS = [
    # Run 1 — Baseline
    {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 5,
     "ent_coef": 0.01, "vf_coef": 0.25,
     "gae_lambda": 1.0, "max_grad_norm": 0.5},
    # Run 2 — Lower LR
    {"learning_rate": 1e-4, "gamma": 0.99, "n_steps": 5,
     "ent_coef": 0.01, "vf_coef": 0.25,
     "gae_lambda": 1.0, "max_grad_norm": 0.5},
    # Run 3 — Higher LR
    {"learning_rate": 1e-3, "gamma": 0.99, "n_steps": 5,
     "ent_coef": 0.01, "vf_coef": 0.25,
     "gae_lambda": 1.0, "max_grad_norm": 0.5},
    # Run 4 — More steps per update (longer rollout)
    {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 20,
     "ent_coef": 0.01, "vf_coef": 0.25,
     "gae_lambda": 1.0, "max_grad_norm": 0.5},
    # Run 5 — Even longer rollout
    {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 50,
     "ent_coef": 0.01, "vf_coef": 0.25,
     "gae_lambda": 1.0, "max_grad_norm": 0.5},
    # Run 6 — High entropy coeff (explore more)
    {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 5,
     "ent_coef": 0.05, "vf_coef": 0.25,
     "gae_lambda": 1.0, "max_grad_norm": 0.5},
    # Run 7 — Higher value function weight
    {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 5,
     "ent_coef": 0.01, "vf_coef": 0.5,
     "gae_lambda": 1.0, "max_grad_norm": 0.5},
    # Run 8 — GAE lambda < 1 (bias-variance tradeoff)
    {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 5,
     "ent_coef": 0.01, "vf_coef": 0.25,
     "gae_lambda": 0.95, "max_grad_norm": 0.5},
    # Run 9 — Low gamma
    {"learning_rate": 7e-4, "gamma": 0.85, "n_steps": 5,
     "ent_coef": 0.01, "vf_coef": 0.25,
     "gae_lambda": 1.0, "max_grad_norm": 0.5},
    # Run 10 — Best combination
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 20,
     "ent_coef": 0.02, "vf_coef": 0.4,
     "gae_lambda": 0.95, "max_grad_norm": 0.5},
]


# ═════════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ═════════════════════════════════════════════════════════════════════════

def plot_pg_curves(all_rewards, run_labels, algo_name,
                   all_entropies=None):
    """
    Plot learning curves and entropy curves for a policy gradient algorithm.
    Saves two PNG files: one for rewards, one for entropy.
    """
    colours = plt.cm.tab10(np.linspace(0, 1, len(all_rewards)))
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    fig.suptitle(
        f"{algo_name.upper()} — 10 Hyperparameter Runs "
        f"(PharmacySupplyEnv)",
        fontsize=14, fontweight="bold"
    )

    # ── Top: reward curves ────────────────────────────────────────────────
    ax1 = axes[0]
    for i, (rewards, label) in enumerate(zip(all_rewards, run_labels)):
        if len(rewards) < 10:
            continue
        window = min(10, len(rewards))
        smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
        ax1.plot(smoothed, label=label, color=colours[i], linewidth=1.5)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("Cumulative reward per episode (smoothed)")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="black", linewidth=0.5, linestyle="--")

    # ── Bottom: mean reward bar chart ─────────────────────────────────────
    ax2 = axes[1]
    mean_rewards = []
    std_rewards  = []
    for r in all_rewards:
        last = r[-20:] if len(r) >= 20 else r
        mean_rewards.append(np.mean(last) if last else 0)
        std_rewards.append(np.std(last)   if last else 0)

    bars = ax2.bar(run_labels, mean_rewards, color=colours,
                   yerr=std_rewards, capsize=4,
                   edgecolor="black", linewidth=0.5)
    for bar, m in zip(bars, mean_rewards):
        bar.set_color("#50c878" if m >= 0 else "#dc5050")
    ax2.set_xlabel("Run")
    ax2.set_ylabel("Mean Reward (last 20 episodes)")
    ax2.set_title("Final performance comparison")
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    reward_path = os.path.join(PLOT_DIR, f"{algo_name}_reward_curves.png")
    plt.savefig(reward_path, dpi=150, bbox_inches="tight")
    print(f"\nReward curve saved: {reward_path}")
    plt.close()

    # ── Entropy curve plot ────────────────────────────────────────────────
    if all_entropies:
        fig2, ax = plt.subplots(figsize=(12, 4))
        for i, (ents, label) in enumerate(zip(all_entropies, run_labels)):
            if len(ents) < 2:
                continue
            window   = min(10, len(ents))
            smoothed = np.convolve(ents, np.ones(window)/window, mode="valid")
            ax.plot(smoothed, label=label, color=colours[i], linewidth=1.5)
        ax.set_xlabel("Episode / Update step")
        ax.set_ylabel("Policy Entropy")
        ax.set_title(
            f"{algo_name.upper()} — Policy Entropy Over Training\n"
            "Higher entropy = more exploration; "
            "Decreasing entropy = agent becoming more confident"
        )
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        entropy_path = os.path.join(PLOT_DIR, f"{algo_name}_entropy_curves.png")
        plt.savefig(entropy_path, dpi=150, bbox_inches="tight")
        print(f"Entropy curve saved: {entropy_path}")
        plt.close()


def print_results_table(results, algo_name, param_keys):
    """Print a formatted results table to the terminal."""
    print(f"\n{'='*70}")
    print(f"  {algo_name.upper()} RESULTS TABLE")
    print(f"{'='*70}")

    col_w   = 10
    header  = f"{'Run':>4}"
    for k in param_keys:
        header += f"  {k[:col_w]:>{col_w}}"
    header += f"  {'MeanR':>8}  {'StdR':>7}  {'Episodes':>8}"
    print(header)
    print("-" * len(header))

    best = max(results, key=lambda x: x["mean_reward"])

    for r in results:
        row = f"{r['run']:>4}"
        for k in param_keys:
            val = r.get(k, "")
            if isinstance(val, float):
                row += f"  {val:>{col_w}.4g}"
            else:
                row += f"  {str(val):>{col_w}}"
        marker = " ◀ BEST" if r["run"] == best["run"] else ""
        row += (f"  {r['mean_reward']:>8.1f}"
                f"  {r['std_reward']:>7.1f}"
                f"  {r['n_episodes']:>8}{marker}")
        print(row)

    print(f"\n  Best run: Run {best['run']} "
          f"(mean reward = {best['mean_reward']:.1f})")


# ═════════════════════════════════════════════════════════════════════════
# MAIN RUNNERS — one function per algorithm
# ═════════════════════════════════════════════════════════════════════════

def run_reinforce():
    print("\n" + "="*60)
    print("  REINFORCE — 10 Hyperparameter Experiments")
    print("="*60)

    all_rewards  = []
    all_entropies = []
    run_labels   = []
    results      = []

    for run_id, hparams in enumerate(REINFORCE_RUNS, start=1):
        rewards, mean_r, std_r, entropies = train_reinforce(
            run_id=run_id,
            hyperparams=hparams,
            n_episodes=400,
            seed=42,
        )
        all_rewards.append(rewards)
        all_entropies.append(entropies)
        run_labels.append(f"R{run_id}")
        results.append({
            "run": run_id,
            "learning_rate": hparams["learning_rate"],
            "gamma": hparams["gamma"],
            "hidden_size": hparams["hidden_size"],
            "entropy_coeff": hparams["entropy_coeff"],
            "mean_reward": round(mean_r, 2),
            "std_reward": round(std_r, 2),
            "n_episodes": len(rewards),
        })

    plot_pg_curves(all_rewards, run_labels, "reinforce", all_entropies)
    print_results_table(
        results, "REINFORCE",
        ["learning_rate", "gamma", "hidden_size", "entropy_coeff"]
    )
    return results


def run_ppo():
    print("\n" + "="*60)
    print("  PPO — 10 Hyperparameter Experiments")
    print("="*60)

    all_rewards = []
    run_labels  = []
    results     = []

    for run_id, hparams in enumerate(PPO_RUNS, start=1):
        _, callback, mean_r, std_r = train_sb3_algo(
            "ppo", run_id, hparams,
            total_timesteps=80_000, seed=42,
        )
        all_rewards.append(callback.episode_rewards)
        run_labels.append(f"R{run_id}")
        results.append({
            "run": run_id,
            "learning_rate": hparams["learning_rate"],
            "gamma": hparams["gamma"],
            "clip_range": hparams["clip_range"],
            "n_epochs": hparams["n_epochs"],
            "ent_coef": hparams["ent_coef"],
            "n_steps": hparams["n_steps"],
            "mean_reward": round(mean_r, 2),
            "std_reward": round(std_r, 2),
            "n_episodes": len(callback.episode_rewards),
        })

    plot_pg_curves(all_rewards, run_labels, "ppo")
    print_results_table(
        results, "PPO",
        ["learning_rate", "gamma", "clip_range", "n_epochs",
         "ent_coef", "n_steps"]
    )
    return results


def run_a2c():
    print("\n" + "="*60)
    print("  A2C — 10 Hyperparameter Experiments")
    print("="*60)

    all_rewards = []
    run_labels  = []
    results     = []

    for run_id, hparams in enumerate(A2C_RUNS, start=1):
        _, callback, mean_r, std_r = train_sb3_algo(
            "a2c", run_id, hparams,
            total_timesteps=80_000, seed=42,
        )
        all_rewards.append(callback.episode_rewards)
        run_labels.append(f"R{run_id}")
        results.append({
            "run": run_id,
            "learning_rate": hparams["learning_rate"],
            "gamma": hparams["gamma"],
            "n_steps": hparams["n_steps"],
            "ent_coef": hparams["ent_coef"],
            "vf_coef": hparams["vf_coef"],
            "gae_lambda": hparams["gae_lambda"],
            "mean_reward": round(mean_r, 2),
            "std_reward": round(std_r, 2),
            "n_episodes": len(callback.episode_rewards),
        })

    plot_pg_curves(all_rewards, run_labels, "a2c")
    print_results_table(
        results, "A2C",
        ["learning_rate", "gamma", "n_steps", "ent_coef",
         "vf_coef", "gae_lambda"]
    )
    return results


# ═════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train policy gradient RL agents on PharmacySupplyEnv"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="all",
        choices=["all", "reinforce", "ppo", "a2c"],
        help="Which algorithm to train (default: all)"
    )
    args = parser.parse_args()

    if args.algo in ("all", "reinforce"):
        run_reinforce()
    if args.algo in ("all", "ppo"):
        run_ppo()
    if args.algo in ("all", "a2c"):
        run_a2c()

    print("\n" + "="*60)
    print("  ALL POLICY GRADIENT TRAINING COMPLETE")
    print("  Plots saved to: models/pg/plots/")
    print("  Models saved to: models/pg/reinforce|ppo|a2c/")
    print("="*60)


if __name__ == "__main__":
    main()