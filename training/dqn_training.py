"""
dqn_training.py
===============
Trains a Deep Q-Network (DQN) agent on the PharmacySupplyEnv-v0
environment using Stable Baselines 3.

What is DQN?
------------
DQN (Deep Q-Network) is a VALUE-BASED reinforcement learning algorithm.
Instead of directly learning what action to take, it learns to estimate
the VALUE of every possible action at each state, then picks the best.

  DQN  : "In this state, action A is worth 80 pts, action B 30 pts → pick A"

Two key mechanisms:
  - Replay buffer : stores past (state, action, reward, next_state) tuples
                    and samples randomly from them to break correlation
  - Target network: a slowly-updated copy of the Q-network that stabilises
                    training by providing consistent update targets

Why the agent was doing nothing before:
  The untrained (or barely trained) DQN outputs nearly equal Q-values for
  all actions, so action 0 (do nothing) often wins by tiny margin.
  Fix: train for more timesteps with lower learning_starts so the network
  sees real consequences of inaction before it gets stuck.

Usage:
    python training/dqn_training.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

from environment.custom_env import PharmacySupplyEnv

MODEL_DIR = os.path.join("models", "dqn")
PLOT_DIR  = os.path.join("models", "dqn", "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────
class RewardLoggerCallback(BaseCallback):
    """Records episode rewards and lengths during training."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards  = []
        self.episode_lengths  = []
        self._current_rewards = []

    def _on_step(self) -> bool:
        self._current_rewards.append(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self.episode_rewards.append(sum(self._current_rewards))
            self.episode_lengths.append(len(self._current_rewards))
            self._current_rewards = []
        return True


# ─────────────────────────────────────────────────────────────────────────
def train_dqn(run_id, hyperparams, total_timesteps=150_000, seed=42):
    """
    Train one DQN model with given hyperparameters.

    150,000 timesteps (up from 80k) ensures the agent has enough
    experience to learn that NOT ordering leads to stockout penalties.

    Returns: model, callback, mean_reward, std_reward
    """
    print(f"\n{'='*62}")
    print(f"  DQN Run {run_id:02d} | "
          f"LR={hyperparams['learning_rate']:.0e} | "
          f"gamma={hyperparams['gamma']} | "
          f"buffer={hyperparams['buffer_size']:,} | "
          f"explr={hyperparams['exploration_fraction']}")
    print(f"{'='*62}")

    env      = Monitor(PharmacySupplyEnv(seed=seed))
    eval_env = Monitor(PharmacySupplyEnv(seed=seed + 100))

    reward_cb = RewardLoggerCallback()
    eval_cb   = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_DIR, f"run_{run_id:02d}_best"),
        log_path=os.path.join(MODEL_DIR, f"run_{run_id:02d}_logs"),
        eval_freq=15_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=0,
    )

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=hyperparams["learning_rate"],
        gamma=hyperparams["gamma"],
        buffer_size=hyperparams["buffer_size"],
        batch_size=hyperparams["batch_size"],
        exploration_fraction=hyperparams["exploration_fraction"],
        exploration_final_eps=hyperparams["exploration_final_eps"],
        target_update_interval=hyperparams["target_update_interval"],
        learning_starts=hyperparams["learning_starts"],
        train_freq=hyperparams["train_freq"],
        policy_kwargs=dict(net_arch=[128, 128]),  # bigger network
        verbose=1,
        seed=seed,
        device="auto",
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[reward_cb, eval_cb],
        progress_bar=True,
    )

    last_20     = reward_cb.episode_rewards[-20:]
    mean_reward = np.mean(last_20) if last_20 else 0.0
    std_reward  = np.std(last_20)  if last_20 else 0.0

    save_path = os.path.join(MODEL_DIR, f"dqn_run_{run_id:02d}")
    model.save(save_path)
    print(f"\n  Run {run_id:02d} | Mean reward (last 20 eps): "
          f"{mean_reward:.1f} ± {std_reward:.1f}")
    print(f"  Saved: {save_path}.zip")

    env.close()
    eval_env.close()
    return model, reward_cb, mean_reward, std_reward


# ─────────────────────────────────────────────────────────────────────────
def plot_learning_curves(all_rewards, run_labels):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    fig.suptitle("DQN — 10 Hyperparameter Runs (PharmacySupplyEnv)",
                 fontsize=14, fontweight="bold")
    colours = plt.cm.tab10(np.linspace(0, 1, len(all_rewards)))

    ax1 = axes[0]
    for i, (rewards, label) in enumerate(zip(all_rewards, run_labels)):
        if len(rewards) < 10:
            continue
        smoothed = np.convolve(rewards, np.ones(10)/10, mode="valid")
        ax1.plot(smoothed, label=label, color=colours[i], linewidth=1.5)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("Episode reward per run (smoothed, window=10)")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")

    ax2 = axes[1]
    means = [np.mean(r[-20:]) if len(r) >= 20 else np.mean(r) if r else 0
             for r in all_rewards]
    stds  = [np.std(r[-20:])  if len(r) >= 20 else np.std(r)  if r else 0
             for r in all_rewards]
    bars = ax2.bar(run_labels, means, color=colours,
                   yerr=stds, capsize=4, edgecolor="black", linewidth=0.5)
    for bar, m in zip(bars, means):
        bar.set_color("#50c878" if m >= 0 else "#dc5050")
    ax2.set_xlabel("Run")
    ax2.set_ylabel("Mean Reward (last 20 episodes)")
    ax2.set_title("Final performance comparison across runs")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, "dqn_learning_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nLearning curves saved: {path}")
    plt.close()


def plot_objective_curve(run_id):
    """Plot evaluation reward curve from EvalCallback logs."""
    log_path = os.path.join(MODEL_DIR, f"run_{run_id:02d}_logs",
                            "evaluations.npz")
    if not os.path.exists(log_path):
        return

    data      = np.load(log_path)
    timesteps = data["timesteps"]
    results   = data["results"]
    means     = results.mean(axis=1)
    stds      = results.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timesteps, means, color="#50a0f0", linewidth=2,
            label="Mean eval reward")
    ax.fill_between(timesteps, means - stds, means + stds,
                    alpha=0.2, color="#50a0f0", label="±1 std")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean evaluation reward")
    ax.set_title(f"DQN Objective (Q-loss proxy) — Run {run_id:02d}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"dqn_objective_run_{run_id:02d}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Objective curve: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────
# 10 HYPERPARAMETER CONFIGURATIONS
# Key difference from v1: learning_starts is lower (500 instead of 1000)
# so the agent starts updating earlier and breaks the do-nothing habit.
# ─────────────────────────────────────────────────────────────────────────
HYPERPARAMETER_RUNS = [
    # Run 1 — Baseline (well-tuned starting point)
    dict(learning_rate=1e-3, gamma=0.99, buffer_size=50_000,
         batch_size=64,  exploration_fraction=0.3,
         exploration_final_eps=0.05, target_update_interval=500,
         learning_starts=500, train_freq=4),

    # Run 2 — Lower LR (slower, more stable convergence)
    dict(learning_rate=1e-4, gamma=0.99, buffer_size=50_000,
         batch_size=64,  exploration_fraction=0.3,
         exploration_final_eps=0.05, target_update_interval=500,
         learning_starts=500, train_freq=4),

    # Run 3 — Higher LR (faster but risks instability)
    dict(learning_rate=5e-3, gamma=0.99, buffer_size=50_000,
         batch_size=64,  exploration_fraction=0.3,
         exploration_final_eps=0.05, target_update_interval=500,
         learning_starts=500, train_freq=4),

    # Run 4 — Lower gamma (agent values immediate rewards more)
    dict(learning_rate=1e-3, gamma=0.90, buffer_size=50_000,
         batch_size=64,  exploration_fraction=0.3,
         exploration_final_eps=0.05, target_update_interval=500,
         learning_starts=500, train_freq=4),

    # Run 5 — Very low gamma (very short-sighted agent)
    dict(learning_rate=1e-3, gamma=0.70, buffer_size=50_000,
         batch_size=64,  exploration_fraction=0.3,
         exploration_final_eps=0.05, target_update_interval=500,
         learning_starts=500, train_freq=4),

    # Run 6 — Large buffer + large batch (diverse replay)
    dict(learning_rate=1e-3, gamma=0.99, buffer_size=200_000,
         batch_size=128, exploration_fraction=0.3,
         exploration_final_eps=0.05, target_update_interval=500,
         learning_starts=1_000, train_freq=4),

    # Run 7 — Small buffer (faster turnover, less memory)
    dict(learning_rate=1e-3, gamma=0.99, buffer_size=10_000,
         batch_size=32,  exploration_fraction=0.3,
         exploration_final_eps=0.05, target_update_interval=500,
         learning_starts=200, train_freq=4),

    # Run 8 — More exploration (stays random longer before exploiting)
    dict(learning_rate=1e-3, gamma=0.99, buffer_size=50_000,
         batch_size=64,  exploration_fraction=0.6,
         exploration_final_eps=0.10, target_update_interval=500,
         learning_starts=500, train_freq=4),

    # Run 9 — Less exploration (exploits knowledge sooner)
    dict(learning_rate=1e-3, gamma=0.99, buffer_size=50_000,
         batch_size=64,  exploration_fraction=0.1,
         exploration_final_eps=0.01, target_update_interval=500,
         learning_starts=500, train_freq=4),

    # Run 10 — Best combination from above analysis
    dict(learning_rate=1e-4, gamma=0.99, buffer_size=100_000,
         batch_size=128, exploration_fraction=0.4,
         exploration_final_eps=0.05, target_update_interval=250,
         learning_starts=500, train_freq=2),
]


# ─────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*62)
    print("  DQN HYPERPARAMETER EXPERIMENTS")
    print("  PharmacySupplyEnv-v0 — Medication Shortage Prevention")
    print("="*62)

    all_rewards  = []
    run_labels   = []
    results_rows = []

    for run_id, hparams in enumerate(HYPERPARAMETER_RUNS, start=1):
        model, cb, mean_r, std_r = train_dqn(
            run_id=run_id,
            hyperparams=hparams,
            total_timesteps=150_000,
            seed=42,
        )
        all_rewards.append(cb.episode_rewards)
        run_labels.append(f"R{run_id}")
        results_rows.append(dict(
            run=run_id,
            learning_rate=hparams["learning_rate"],
            gamma=hparams["gamma"],
            buffer_size=hparams["buffer_size"],
            batch_size=hparams["batch_size"],
            exploration_fraction=hparams["exploration_fraction"],
            target_update_interval=hparams["target_update_interval"],
            mean_reward=round(mean_r, 2),
            std_reward=round(std_r, 2),
            episodes=len(cb.episode_rewards),
        ))
        plot_objective_curve(run_id)

    plot_learning_curves(all_rewards, run_labels)

    # Print results table
    print("\n" + "="*90)
    print("  DQN RESULTS TABLE")
    print("="*90)
    hdr = (f"{'Run':>4} | {'LR':>8} | {'Gamma':>5} | {'Buffer':>8} | "
           f"{'Batch':>5} | {'Explr':>5} | {'TgtUpd':>6} | "
           f"{'MeanR':>8} | {'StdR':>7} | {'Eps':>5}")
    print(hdr)
    print("-" * len(hdr))

    best = max(results_rows, key=lambda x: x["mean_reward"])
    for r in results_rows:
        flag = " ◀ BEST" if r["run"] == best["run"] else ""
        print(
            f"{r['run']:>4} | {r['learning_rate']:>8.0e} | "
            f"{r['gamma']:>5.2f} | {r['buffer_size']:>8,} | "
            f"{r['batch_size']:>5} | {r['exploration_fraction']:>5.1f} | "
            f"{r['target_update_interval']:>6} | "
            f"{r['mean_reward']:>8.1f} | {r['std_reward']:>7.1f} | "
            f"{r['episodes']:>5}{flag}"
        )

    print(f"\n  Best run: Run {best['run']} "
          f"| Mean reward = {best['mean_reward']:.1f}")
    print(f"  Best model: models/dqn/dqn_run_{best['run']:02d}.zip")
    print("\nDQN training complete!")


if __name__ == "__main__":
    main()
    