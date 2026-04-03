"""
dqn_training.py
===============
Trains a Deep Q-Network (DQN) agent on the PharmacySupplyEnv-v0
environment using Stable Baselines 3.

What is DQN?
------------
DQN (Deep Q-Network) is a VALUE-BASED reinforcement learning algorithm.
Instead of directly learning "what action to take", it learns to estimate
the VALUE of every possible action at each state — then picks the best one.

Think of it like this:
  - The agent builds a "cheat sheet" (Q-table) that says:
    "If I'm in situation X, action A is worth 50 points, action B is worth 30..."
  - A neural network approximates this cheat sheet for complex environments
  - Over time the estimates get more accurate and the agent improves

Key hyperparameters we tune:
  - learning_rate   : how fast the network updates its weights
  - gamma           : how much the agent cares about future rewards (0=now, 1=far future)
  - buffer_size     : how many past experiences to remember and learn from
  - batch_size      : how many memories to sample per training update
  - exploration_fraction: how long to explore randomly before exploiting knowledge
  - target_update_interval: how often to sync the target network

Usage:
    python training/dqn_training.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves plots to file

# ── Path setup: allow imports from project root ───────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import PharmacySupplyEnv

# ── Output directories ────────────────────────────────────────────────────
MODEL_DIR  = os.path.join("models", "dqn")
PLOT_DIR   = os.path.join("models", "dqn", "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────
# REWARD LOGGER CALLBACK
# Records episode rewards during training so we can plot them later
# ─────────────────────────────────────────────────────────────────────────
class RewardLoggerCallback(BaseCallback):
    """
    A custom callback that records the reward at the end of each episode.
    We use this data to plot learning curves after training.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards  = []
        self.episode_lengths  = []
        self._current_rewards = []

    def _on_step(self) -> bool:
        # self.locals["rewards"] is a list (one entry per env in vec env)
        reward = self.locals["rewards"][0]
        self._current_rewards.append(reward)

        # Check if episode ended
        done = self.locals["dones"][0]
        if done:
            total = sum(self._current_rewards)
            self.episode_rewards.append(total)
            self.episode_lengths.append(len(self._current_rewards))
            self._current_rewards = []

        return True  # returning False would stop training


# ─────────────────────────────────────────────────────────────────────────
# TRAIN ONE DQN MODEL
# ─────────────────────────────────────────────────────────────────────────
def train_dqn(run_id, hyperparams, total_timesteps=80_000, seed=42):
    """
    Train a single DQN model with the given hyperparameters.

    Args:
        run_id         : integer label for this run (1-10)
        hyperparams    : dict of DQN hyperparameters to use
        total_timesteps: how many environment steps to train for
        seed           : random seed for reproducibility

    Returns:
        model          : the trained SB3 DQN model
        callback       : the reward logger (contains episode_rewards list)
        mean_reward    : average reward over the last 20 episodes
    """
    print(f"\n{'='*60}")
    print(f"  DQN Run {run_id:02d} | LR={hyperparams['learning_rate']} | "
          f"gamma={hyperparams['gamma']} | "
          f"buffer={hyperparams['buffer_size']}")
    print(f"{'='*60}")

    # ── Create the environment ────────────────────────────────────────────
    # Monitor wraps the env to log episode stats automatically
    env = Monitor(PharmacySupplyEnv(seed=seed))

    # Evaluation environment (separate from training env — good practice)
    eval_env = Monitor(PharmacySupplyEnv(seed=seed + 100))

    # ── Reward logger callback ────────────────────────────────────────────
    reward_callback = RewardLoggerCallback()

    # ── EvalCallback: evaluates the model every N steps ──────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_DIR, f"run_{run_id:02d}_best"),
        log_path=os.path.join(MODEL_DIR, f"run_{run_id:02d}_logs"),
        eval_freq=10_000,           # evaluate every 10,000 steps
        n_eval_episodes=5,          # run 5 evaluation episodes
        deterministic=True,         # no exploration during evaluation
        render=False,
        verbose=0,
    )

    # ── Build the DQN model ───────────────────────────────────────────────
    model = DQN(
        policy="MlpPolicy",         # Multi-layer perceptron (standard feedforward NN)
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
        verbose=1,
        seed=seed,
        device="auto",              # uses GPU if available, else CPU
    )

    # ── Train the model ───────────────────────────────────────────────────
    model.learn(
        total_timesteps=total_timesteps,
        callback=[reward_callback, eval_callback],
        progress_bar=True,
    )

    # ── Calculate final performance ───────────────────────────────────────
    last_20 = reward_callback.episode_rewards[-20:]
    mean_reward = np.mean(last_20) if last_20 else 0.0
    std_reward  = np.std(last_20)  if last_20 else 0.0

    print(f"\n  Run {run_id:02d} complete.")
    print(f"  Mean reward (last 20 episodes): {mean_reward:.1f} ± {std_reward:.1f}")

    # ── Save the model ────────────────────────────────────────────────────
    save_path = os.path.join(MODEL_DIR, f"dqn_run_{run_id:02d}")
    model.save(save_path)
    print(f"  Model saved to: {save_path}.zip")

    env.close()
    eval_env.close()

    return model, reward_callback, mean_reward, std_reward


# ─────────────────────────────────────────────────────────────────────────
# PLOT LEARNING CURVES
# ─────────────────────────────────────────────────────────────────────────
def plot_learning_curves(all_rewards, run_labels, title="DQN Learning Curves"):
    """
    Plot episode reward curves for all runs on one figure.
    Saves the plot as a PNG file.

    Args:
        all_rewards : list of lists — episode rewards per run
        run_labels  : list of strings — label for each run
        title       : plot title
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_rewards)))

    # ── Top plot: individual reward curves ───────────────────────────────
    ax1 = axes[0]
    for i, (rewards, label) in enumerate(zip(all_rewards, run_labels)):
        if not rewards:
            continue
        # Smooth curve with a rolling window of 10 episodes
        smoothed = np.convolve(rewards, np.ones(10)/10, mode="valid")
        ax1.plot(smoothed, label=label, color=colors[i], linewidth=1.5)

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("Episode reward per run (smoothed, window=10)")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="black", linewidth=0.5, linestyle="--")

    # ── Bottom plot: final mean reward bar chart ──────────────────────────
    ax2 = axes[1]
    mean_rewards = [np.mean(r[-20:]) if len(r) >= 20 else np.mean(r)
                    for r in all_rewards]
    std_rewards  = [np.std(r[-20:])  if len(r) >= 20 else np.std(r)
                    for r in all_rewards]

    bars = ax2.bar(run_labels, mean_rewards, color=colors,
                   yerr=std_rewards, capsize=4, edgecolor="black", linewidth=0.5)

    # Colour bars: green if positive, red if negative
    for bar, mean in zip(bars, mean_rewards):
        bar.set_color("#50c878" if mean >= 0 else "#dc5050")

    ax2.set_xlabel("Run")
    ax2.set_ylabel("Mean Reward (last 20 episodes)")
    ax2.set_title("Final performance comparison across hyperparameter runs")
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, "dqn_learning_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nLearning curve saved to: {save_path}")
    plt.close()


def plot_loss_curve(model_path, run_id):
    """
    Plot the DQN training loss (TD error / objective curve).
    SB3 logs this to tensorboard — we read the monitor CSV instead.
    """
    # SB3 Monitor logs episode data to a .monitor.csv file
    # For loss we use the eval callback logs
    log_path = os.path.join(MODEL_DIR, f"run_{run_id:02d}_logs", "evaluations.npz")
    if not os.path.exists(log_path):
        print(f"  No eval log found for run {run_id}. Skipping loss plot.")
        return

    data = np.load(log_path)
    timesteps = data["timesteps"]
    results   = data["results"]           # shape: (n_evals, n_eval_episodes)
    mean_eval = results.mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timesteps, mean_eval, color="#50a0f0", linewidth=2)
    ax.fill_between(timesteps,
                    mean_eval - results.std(axis=1),
                    mean_eval + results.std(axis=1),
                    alpha=0.2, color="#50a0f0")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean evaluation reward")
    ax.set_title(f"DQN Objective Curve — Run {run_id:02d} (eval every 10k steps)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, f"dqn_objective_run_{run_id:02d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Objective curve saved to: {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────
# 10 HYPERPARAMETER CONFIGURATIONS
# Each row in this list corresponds to one run in the hyperparameter table.
# We vary: learning_rate, gamma, buffer_size, batch_size,
#          exploration_fraction, target_update_interval
# ─────────────────────────────────────────────────────────────────────────
HYPERPARAMETER_RUNS = [
    # Run 1 — Baseline (default-ish settings)
    {
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "buffer_size": 50_000,
        "batch_size": 64,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
        "learning_starts": 1_000,
        "train_freq": 4,
    },
    # Run 2 — Lower learning rate (slower, more stable learning)
    {
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "buffer_size": 50_000,
        "batch_size": 64,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
        "learning_starts": 1_000,
        "train_freq": 4,
    },
    # Run 3 — Higher learning rate (faster but potentially unstable)
    {
        "learning_rate": 5e-3,
        "gamma": 0.99,
        "buffer_size": 50_000,
        "batch_size": 64,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
        "learning_starts": 1_000,
        "train_freq": 4,
    },
    # Run 4 — Lower gamma (agent values immediate rewards more)
    {
        "learning_rate": 1e-3,
        "gamma": 0.90,
        "buffer_size": 50_000,
        "batch_size": 64,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
        "learning_starts": 1_000,
        "train_freq": 4,
    },
    # Run 5 — Very low gamma (very short-sighted agent)
    {
        "learning_rate": 1e-3,
        "gamma": 0.70,
        "buffer_size": 50_000,
        "batch_size": 64,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
        "learning_starts": 1_000,
        "train_freq": 4,
    },
    # Run 6 — Large replay buffer (more diverse past experiences)
    {
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "buffer_size": 200_000,
        "batch_size": 128,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
        "learning_starts": 2_000,
        "train_freq": 4,
    },
    # Run 7 — Small replay buffer (less memory, faster updates)
    {
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "buffer_size": 10_000,
        "batch_size": 32,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
        "learning_starts": 500,
        "train_freq": 4,
    },
    # Run 8 — More exploration (explores randomly for longer)
    {
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "buffer_size": 50_000,
        "batch_size": 64,
        "exploration_fraction": 0.6,
        "exploration_final_eps": 0.10,
        "target_update_interval": 500,
        "learning_starts": 1_000,
        "train_freq": 4,
    },
    # Run 9 — Less exploration (exploits knowledge sooner)
    {
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "buffer_size": 50_000,
        "batch_size": 64,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.01,
        "target_update_interval": 500,
        "learning_starts": 1_000,
        "train_freq": 4,
    },
    # Run 10 — Frequent target updates + best settings from above
    {
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "buffer_size": 100_000,
        "batch_size": 128,
        "exploration_fraction": 0.4,
        "exploration_final_eps": 0.05,
        "target_update_interval": 250,
        "learning_starts": 2_000,
        "train_freq": 2,
    },
]


# ─────────────────────────────────────────────────────────────────────────
# MAIN — run all 10 DQN experiments
# ─────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  DQN HYPERPARAMETER EXPERIMENTS")
    print("  PharmacySupplyEnv-v0 — Medication Shortage Prevention")
    print("="*60)

    all_rewards  = []
    run_labels   = []
    results_table = []

    for run_id, hparams in enumerate(HYPERPARAMETER_RUNS, start=1):
        model, callback, mean_r, std_r = train_dqn(
            run_id=run_id,
            hyperparams=hparams,
            total_timesteps=80_000,
            seed=42,
        )

        all_rewards.append(callback.episode_rewards)
        run_labels.append(f"R{run_id}")
        results_table.append({
            "run": run_id,
            "learning_rate": hparams["learning_rate"],
            "gamma": hparams["gamma"],
            "buffer_size": hparams["buffer_size"],
            "batch_size": hparams["batch_size"],
            "exploration_fraction": hparams["exploration_fraction"],
            "target_update_interval": hparams["target_update_interval"],
            "mean_reward": round(mean_r, 2),
            "std_reward": round(std_r, 2),
            "episodes": len(callback.episode_rewards),
        })

        # Plot objective curve for this run
        plot_loss_curve(MODEL_DIR, run_id)

    # ── Plot all learning curves together ─────────────────────────────────
    plot_learning_curves(
        all_rewards, run_labels,
        title="DQN — All 10 Hyperparameter Runs (PharmacySupplyEnv)"
    )

    # ── Print results table ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("  DQN RESULTS SUMMARY TABLE")
    print("="*60)
    header = (f"{'Run':>4} | {'LR':>8} | {'Gamma':>5} | {'Buffer':>8} | "
              f"{'Batch':>5} | {'Explr':>5} | {'TgtUpd':>6} | "
              f"{'MeanR':>8} | {'StdR':>7} | {'Eps':>5}")
    print(header)
    print("-" * len(header))

    best_run = max(results_table, key=lambda x: x["mean_reward"])

    for r in results_table:
        marker = " ◀ BEST" if r["run"] == best_run["run"] else ""
        print(
            f"{r['run']:>4} | {r['learning_rate']:>8.0e} | "
            f"{r['gamma']:>5.2f} | {r['buffer_size']:>8,} | "
            f"{r['batch_size']:>5} | {r['exploration_fraction']:>5.1f} | "
            f"{r['target_update_interval']:>6} | "
            f"{r['mean_reward']:>8.1f} | {r['std_reward']:>7.1f} | "
            f"{r['episodes']:>5}{marker}"
        )

    print(f"\n  Best run: Run {best_run['run']} "
          f"(mean reward = {best_run['mean_reward']:.1f})")

    # Save best model path for main.py to use
    best_path = os.path.join(MODEL_DIR, f"dqn_run_{best_run['run']:02d}")
    print(f"  Best model path: {best_path}.zip")
    print("\nDQN training complete!")


if __name__ == "__main__":
    main()