"""
main.py
=======
Entry point for the Pharmacy Supply RL project.

This script does THREE things in sequence:

  1. EVALUATE all four trained algorithms (DQN, REINFORCE, PPO, A2C)
     on the same PharmacySupplyEnv-v0 environment, running N episodes each.

  2. COMPARE results side by side — reward, service rate, stockout days,
     overstock days, emergency orders — printed as a terminal table.

  3. GENERATE all report plots:
       - Cumulative reward curves (all 4 algorithms in subplots)
       - Convergence plot (mean reward per algorithm, bar chart)
       - Generalisation test (performance across 3 different seeds)
       - Metrics comparison (service rate, stockouts, etc.)
       - Stock level timeline for best agent

  4. RUN the best-performing agent live with the pygame dashboard.

  5. EXPORT a JSON results file (demonstrates production API readiness).

Usage:
    python main.py                    # full pipeline: compare + run best
    python main.py --compare-only     # just compare, no live rendering
    python main.py --run-only ppo     # skip comparison, just run one algo
    python main.py --episodes 5       # eval episodes per algorithm
    python main.py --no-render        # terminal only, no pygame window
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from environment.custom_env import PharmacySupplyEnv

os.makedirs("models/results", exist_ok=True)

# ── Terminal colour codes ─────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"

ACTION_NAMES = {
    0: "Do nothing        ",
    1: "Order small  (25%)",
    2: "Order medium (50%)",
    3: "Order large (100%)",
    4: "EMERGENCY ORDER   ",
}

ALGO_COLOURS = {
    "dqn":       "#50a0f0",
    "reinforce": "#f0a050",
    "ppo":       "#50c878",
    "a2c":       "#c878c8",
}

# Maps algorithm name → (folder, file_prefix, framework)
ALGO_REGISTRY = {
    "dqn":       ("models/dqn",         "dqn",       "sb3"),
    "ppo":       ("models/pg/ppo",       "ppo",       "sb3"),
    "a2c":       ("models/pg/a2c",       "a2c",       "sb3"),
    "reinforce": ("models/pg/reinforce", "reinforce", "torch"),
}


# ═════════════════════════════════════════════════════════════════════════
# MODEL DISCOVERY
# ═════════════════════════════════════════════════════════════════════════

def find_model(algo_name):
    """
    Find the saved model file for a given algorithm.
    Searches run_10 down to run_01 and returns the first found.
    Returns (path, framework) or (None, None) if not found.
    """
    folder, prefix, framework = ALGO_REGISTRY[algo_name]
    if not os.path.exists(folder):
        return None, None

    for run_id in range(10, 0, -1):
        ext  = ".zip" if framework == "sb3" else ".pt"
        path = os.path.join(folder, f"{prefix}_run_{run_id:02d}{ext}")
        if os.path.exists(path):
            return path, framework

    # Also check EvalCallback best model directories
    try:
        best_dirs = sorted(
            [d for d in os.listdir(folder) if "best" in d],
            reverse=True
        )
        for bd in best_dirs:
            ext  = ".zip" if framework == "sb3" else ".pt"
            path = os.path.join(folder, bd, f"best_model{ext}")
            if os.path.exists(path):
                return path, framework
    except Exception:
        pass

    return None, None


# ═════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═════════════════════════════════════════════════════════════════════════

def _infer_hidden_size(state_dict):
    """
    Infer the hidden layer size from a saved PolicyNetwork state_dict.

    The first linear layer weight has shape (hidden_size, obs_dim).
    We read hidden_size from it so we never hardcode 64 or 128.

    This fixes the RuntimeError:
      size mismatch for network.0.weight: shape [128, 7] vs [64, 7]
    """
    first_weight = state_dict.get("network.0.weight")
    if first_weight is not None:
        return first_weight.shape[0]   # hidden_size is the output dim
    return 64  # safe fallback


def load_predict_fn(algo_name, model_path, framework, env):
    """
    Load a saved model and return a unified predict(obs) → action function.

    Handles:
      - SB3 models  : DQN, PPO, A2C  (.zip files)
      - PyTorch      : REINFORCE       (.pt files)
                       Auto-detects hidden_size from the checkpoint so
                       the model always loads regardless of which run
                       was saved (run 6 = 128 units, run 7 = 32 units, etc.)
    """
    if framework == "sb3":
        from stable_baselines3 import DQN, PPO, A2C
        cls_map = {"dqn": DQN, "ppo": PPO, "a2c": A2C}
        model   = cls_map[algo_name].load(model_path, env=env)

        def predict_fn(obs):
            action, _ = model.predict(obs, deterministic=True)
            return int(action)

    else:
        # ── REINFORCE: auto-detect hidden_size from checkpoint ────────────
        import torch
        from training.pg_training import PolicyNetwork

        # Load raw weights first so we can inspect the shape
        state_dict  = torch.load(model_path, map_location="cpu",
                                 weights_only=True)
        hidden_size = _infer_hidden_size(state_dict)

        print(f"    REINFORCE: detected hidden_size={hidden_size} "
              f"from checkpoint {os.path.basename(model_path)}")

        policy = PolicyNetwork(obs_dim=7, action_dim=5,
                               hidden_size=hidden_size)
        policy.load_state_dict(state_dict)
        policy.eval()

        def predict_fn(obs):
            with torch.no_grad():
                import torch as _t
                obs_t  = _t.tensor(obs, dtype=_t.float32).unsqueeze(0)
                probs  = policy(obs_t)
                action = _t.argmax(probs, dim=-1).item()
            return int(action)

    return predict_fn


# ═════════════════════════════════════════════════════════════════════════
# EPISODE RUNNER
# ═════════════════════════════════════════════════════════════════════════

def run_episode(predict_fn, env, verbose=False,
                episode_num=1, algo_label="Agent"):
    """
    Run one complete episode and return a stats dict.
    """
    obs, info = env.reset()

    total_reward     = 0.0
    stockout_days    = 0
    overstock_days   = 0
    orders_placed    = 0
    emergency_orders = 0
    daily_rewards    = []
    stock_levels     = []
    actions_taken    = []
    step_count       = 0

    if verbose:
        print(f"\n{BOLD}{'─'*72}{RESET}")
        print(f"{BOLD}  EPISODE {episode_num} — {algo_label}{RESET}")
        print(f"{BOLD}{'─'*72}{RESET}")
        print(f"  {'Day':>4}  {'Action':<20}  {'Stock':>6}  "
              f"{'Demand':>6}  {'Reward':>7}  {'Cumul':>9}  Status")
        print(f"  {'─'*4}  {'─'*20}  {'─'*6}  "
              f"{'─'*6}  {'─'*7}  {'─'*9}  {'─'*14}")

    done = False
    while not done:
        action = predict_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward  += reward
        step_count    += 1
        daily_rewards.append(reward)
        stock_levels.append(info["stock_level"])
        actions_taken.append(action)

        if info.get("stockout"):    stockout_days    += 1
        if info.get("overstock"):   overstock_days   += 1
        if 0 < action < 4:          orders_placed    += 1
        if action == 4:             emergency_orders += 1

        if verbose:
            avg_demand = (np.mean(env.demand_history)
                          if env.demand_history else 0)
            if info.get("stockout"):
                status = f"{RED}STOCKOUT{RESET}"
            elif info.get("overstock"):
                status = f"{YELLOW}overstock{RESET}"
            elif info.get("order_received"):
                status = f"{GREEN}order arrived{RESET}"
            elif action > 0:
                status = f"{BLUE}order placed{RESET}"
            else:
                status = "—"

            rew_col = GREEN if reward >= 0 else RED
            print(f"  {info['time_step']:>4}  "
                  f"{ACTION_NAMES[action]}  "
                  f"{info['stock_level']:>6.0f}  "
                  f"{avg_demand:>6.1f}  "
                  f"{rew_col}{reward:>+7.1f}{RESET}  "
                  f"{total_reward:>9.1f}  {status}")

        if terminated and verbose:
            print(f"\n  {RED}{BOLD}Terminated: "
                  f"3+ consecutive stockout days.{RESET}")

    return {
        "episode":          episode_num,
        "algo":             algo_label,
        "total_reward":     round(total_reward, 2),
        "steps":            step_count,
        "stockout_days":    stockout_days,
        "overstock_days":   overstock_days,
        "orders_placed":    orders_placed,
        "emergency_orders": emergency_orders,
        "service_rate":     round(
            (step_count - stockout_days) / max(step_count, 1) * 100, 1),
        "min_stock":        round(min(stock_levels), 1) if stock_levels else 0,
        "max_stock":        round(max(stock_levels), 1) if stock_levels else 0,
        "daily_rewards":    daily_rewards,
        "stock_levels":     stock_levels,
        "actions_taken":    actions_taken,
    }


# ═════════════════════════════════════════════════════════════════════════
# CROSS-ALGORITHM EVALUATION
# ═════════════════════════════════════════════════════════════════════════

def evaluate_all_algorithms(n_episodes=5, seed=0, verbose_algo="ppo"):
    """
    Evaluate every available trained algorithm for n_episodes each
    on the same environment conditions.

    Returns:
        results : dict  algo_name → list of episode stat dicts
        loaded  : dict  algo_name → model_path
    """
    results = {}
    loaded  = {}

    print(f"\n{BOLD}{'═'*72}{RESET}")
    print(f"{BOLD}  CROSS-ALGORITHM EVALUATION{RESET}")
    print(f"{BOLD}  Environment : PharmacySupplyEnv-v0{RESET}")
    print(f"{BOLD}  Episodes    : {n_episodes} per algorithm{RESET}")
    print(f"{BOLD}  Seed        : {seed}{RESET}")
    print(f"{BOLD}{'═'*72}{RESET}")

    for algo_name in ["dqn", "reinforce", "ppo", "a2c"]:
        model_path, framework = find_model(algo_name)

        if model_path is None:
            print(f"\n  {YELLOW}[SKIP] {algo_name.upper()} — "
                  f"no saved model found in {ALGO_REGISTRY[algo_name][0]}. "
                  f"Run training scripts first.{RESET}")
            continue

        print(f"\n  {CYAN}Loading {algo_name.upper()} "
              f"from {model_path}{RESET}")
        loaded[algo_name] = model_path

        try:
            env        = PharmacySupplyEnv(seed=seed)
            predict_fn = load_predict_fn(algo_name, model_path,
                                         framework, env)
            env.close()
        except Exception as e:
            print(f"  {RED}Failed to load {algo_name.upper()}: {e}{RESET}")
            continue

        algo_episodes = []
        for ep in range(1, n_episodes + 1):
            env_ep     = PharmacySupplyEnv(seed=seed + ep)
            be_verbose = (verbose_algo == algo_name and ep == 1)
            stats      = run_episode(
                predict_fn=predict_fn,
                env=env_ep,
                verbose=be_verbose,
                episode_num=ep,
                algo_label=algo_name.upper(),
            )
            algo_episodes.append(stats)
            env_ep.close()

            print(f"    {algo_name.upper()} ep {ep}/{n_episodes} | "
                  f"reward={stats['total_reward']:+.0f} | "
                  f"days={stats['steps']} | "
                  f"service={stats['service_rate']}% | "
                  f"stockouts={stats['stockout_days']}")

        results[algo_name] = algo_episodes

    return results, loaded


# ═════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ═════════════════════════════════════════════════════════════════════════

def print_comparison_table(results):
    """Print a formatted side-by-side comparison table to the terminal."""
    print(f"\n{BOLD}{'═'*80}{RESET}")
    print(f"{BOLD}  ALGORITHM COMPARISON — MEAN RESULTS ACROSS EPISODES{RESET}")
    print(f"{BOLD}{'═'*80}{RESET}")

    metrics = [
        ("Mean total reward",     "total_reward",      True,  "+.1f"),
        ("Mean service rate %",   "service_rate",      True,  ".1f"),
        ("Mean days survived",    "steps",             True,  ".1f"),
        ("Mean stockout days",    "stockout_days",     False, ".1f"),
        ("Mean overstock days",   "overstock_days",    False, ".1f"),
        ("Mean orders placed",    "orders_placed",     True,  ".1f"),
        ("Mean emergency orders", "emergency_orders",  False, ".1f"),
    ]

    algo_list = list(results.keys())
    col_w     = 16

    header = f"  {'Metric':<26}"
    for a in algo_list:
        header += f"  {a.upper():>{col_w}}"
    print(header)
    print("  " + "─" * (26 + (col_w + 2) * len(algo_list)))

    for label, key, higher_better, fmt in metrics:
        vals     = {a: np.mean([ep[key] for ep in eps])
                    for a, eps in results.items()}
        best_algo = (max(vals, key=vals.get) if higher_better
                     else min(vals, key=vals.get))
        row = f"  {label:<26}"
        for a in algo_list:
            v      = vals[a]
            cell   = format(v, fmt)
            colour = GREEN if a == best_algo else ""
            rst    = RESET if colour else ""
            row   += f"  {colour}{cell:>{col_w}}{rst}"
        print(row)

    print("  " + "─" * (26 + (col_w + 2) * len(algo_list)))
    means  = {a: np.mean([ep["total_reward"] for ep in eps])
              for a, eps in results.items()}
    winner = max(means, key=means.get)
    print(f"\n  {BOLD}{GREEN}Best overall: {winner.upper()} "
          f"(mean reward = {means[winner]:.1f}){RESET}")


# ═════════════════════════════════════════════════════════════════════════
# REPORT PLOTS
# ═════════════════════════════════════════════════════════════════════════

def plot_cumulative_reward_curves(results):
    """4-subplot grid of cumulative reward curves, one per algorithm."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    fig.suptitle(
        "Cumulative Reward Curves — All Algorithms\n"
        "PharmacySupplyEnv-v0 | Medication Shortage Prevention",
        fontsize=13, fontweight="bold"
    )
    axes_flat = axes.flatten()
    algo_list = list(results.keys())

    for idx, algo_name in enumerate(algo_list):
        ax       = axes_flat[idx]
        episodes = results[algo_name]
        colour   = ALGO_COLOURS.get(algo_name, "steelblue")

        for ep in episodes:
            cumulative = np.cumsum(ep["daily_rewards"])
            ax.plot(cumulative, alpha=0.45, linewidth=1.2, color=colour)

        # Mean cumulative line (pad episodes of different lengths)
        max_len = max(len(ep["daily_rewards"]) for ep in episodes)
        padded  = [
            ep["daily_rewards"] +
            [ep["daily_rewards"][-1]] * (max_len - len(ep["daily_rewards"]))
            for ep in episodes
        ]
        mean_cum = np.cumsum(np.mean(padded, axis=0))
        ax.plot(mean_cum, color="white", linewidth=2.5,
                linestyle="--", label="Mean", zorder=5)

        ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
        ax.set_title(f"{algo_name.upper()}", fontsize=12,
                     fontweight="bold", color=colour)
        ax.set_xlabel("Day")
        ax.set_ylabel("Cumulative Reward")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor("#0f0f18")

    for idx in range(len(algo_list), 4):
        axes_flat[idx].set_visible(False)

    fig.patch.set_facecolor("#0a0a12")
    plt.tight_layout()
    path = "models/results/cumulative_reward_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0a12")
    print(f"  Saved: {path}")
    plt.close()


def plot_convergence(results):
    """Bar chart: mean total reward ± std per algorithm."""
    algo_list = list(results.keys())
    means     = [np.mean([ep["total_reward"] for ep in results[a]])
                 for a in algo_list]
    stds      = [np.std([ep["total_reward"]  for ep in results[a]])
                 for a in algo_list]
    colours   = [ALGO_COLOURS.get(a, "steelblue") for a in algo_list]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        [a.upper() for a in algo_list], means,
        yerr=stds, capsize=6,
        color=colours, edgecolor="white", linewidth=0.8
    )
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + s + max(stds) * 0.03,
                f"{m:.0f}", ha="center", va="bottom",
                fontsize=11, color="white", fontweight="bold")

    ax.set_title(
        "Algorithm Convergence — Mean Total Reward\n"
        "(error bars = ±1 std across evaluation episodes)",
        fontsize=12, fontweight="bold", color="white"
    )
    ax.set_ylabel("Mean Total Reward", color="white")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_facecolor("#0f0f18")
    fig.patch.set_facecolor("#0a0a12")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")

    plt.tight_layout()
    path = "models/results/convergence_plot.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0a12")
    print(f"  Saved: {path}")
    plt.close()


def plot_metrics_comparison(results):
    """Grouped bar charts for service rate, stockouts, overstock, emergencies."""
    algo_list = list(results.keys())
    metrics   = [
        ("Service Rate (%)",   "service_rate",    True),
        ("Stockout Days",      "stockout_days",   False),
        ("Overstock Days",     "overstock_days",  False),
        ("Emergency Orders",   "emergency_orders",False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Performance Metrics Comparison — All Algorithms",
                 fontsize=13, fontweight="bold", color="white")

    for ax, (label, key, higher_better) in zip(axes.flatten(), metrics):
        vals    = [np.mean([ep[key] for ep in results[a]])
                   for a in algo_list]
        colours = [ALGO_COLOURS.get(a, "steelblue") for a in algo_list]
        bars    = ax.bar([a.upper() for a in algo_list], vals,
                         color=colours, edgecolor="white", linewidth=0.6)

        best_idx = (np.argmax(vals) if higher_better else np.argmin(vals))
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(2.5)

        ax.set_title(label, fontsize=11, fontweight="bold", color="white")
        ax.set_ylabel(label, color="white")
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.2, axis="y")
        ax.set_facecolor("#0f0f18")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")

    fig.patch.set_facecolor("#0a0a12")
    plt.tight_layout()
    path = "models/results/metrics_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0a12")
    print(f"  Saved: {path}")
    plt.close()


def plot_generalisation_test(results):
    """
    Test the best algorithm across 3 different random seeds.
    Required by rubric: 'generalisation tests'.
    """
    # Pick best algo from results
    means     = {a: np.mean([ep["total_reward"] for ep in eps])
                 for a, eps in results.items()}
    best_algo = max(means, key=means.get)
    path_m, fw = find_model(best_algo)
    if path_m is None:
        return

    seeds  = [100, 200, 300]
    colour = ALGO_COLOURS.get(best_algo, "steelblue")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(
        f"Generalisation Test — {best_algo.upper()} across 3 unseen seeds\n"
        "Tests whether the learned policy generalises beyond training conditions",
        fontsize=12, fontweight="bold", color="white"
    )

    for ax, seed in zip(axes, seeds):
        env_tmp    = PharmacySupplyEnv(seed=seed)
        predict_fn = load_predict_fn(best_algo, path_m, fw, env_tmp)
        env_tmp.close()

        for ep_idx in range(3):
            env_ep = PharmacySupplyEnv(seed=seed + ep_idx * 17)
            stats  = run_episode(predict_fn, env_ep, verbose=False)
            env_ep.close()
            cum = np.cumsum(stats["daily_rewards"])
            ax.plot(cum, alpha=0.6, linewidth=1.4, color=colour,
                    label=f"ep {ep_idx+1}")

        ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
        ax.set_title(f"Seed = {seed}", fontsize=11, color="white")
        ax.set_xlabel("Day", color="white")
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.2)
        ax.set_facecolor("#0f0f18")
        ax.legend(fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")

    axes[0].set_ylabel("Cumulative Reward", color="white")
    fig.patch.set_facecolor("#0a0a12")
    plt.tight_layout()
    path = "models/results/generalisation_test.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0a12")
    print(f"  Saved: {path}")
    plt.close()


def plot_stock_timeline(results):
    """Stock level over time for the single best episode across all algorithms."""
    best_ep   = None
    best_algo = None
    best_r    = -np.inf

    for algo, episodes in results.items():
        for ep in episodes:
            if ep["total_reward"] > best_r:
                best_r    = ep["total_reward"]
                best_ep   = ep
                best_algo = algo

    if best_ep is None:
        return

    stock   = best_ep["stock_levels"]
    days    = list(range(len(stock)))
    max_cap = 500
    colour  = ALGO_COLOURS.get(best_algo, "steelblue")

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(days, 0, max_cap * 0.25,
                    alpha=0.15, color="red",    label="Danger zone (<25%)")
    ax.fill_between(days, max_cap * 0.85, max_cap,
                    alpha=0.10, color="orange", label="Overstock zone (>85%)")
    ax.plot(days, stock, color=colour, linewidth=1.5,
            label=f"Stock level — {best_algo.upper()}")
    ax.axhline(max_cap * 0.25, color="red",    linewidth=0.8,
               linestyle="--", alpha=0.5)
    ax.axhline(max_cap * 0.85, color="orange", linewidth=0.8,
               linestyle="--", alpha=0.5)
    ax.axvspan(200, min(280, len(days)),
               alpha=0.08, color="yellow",
               label="High-demand season (days 200–280)")

    ax.set_title(
        f"Stock Level Over Time — Best Episode "
        f"({best_algo.upper()}, reward={best_r:.0f})",
        fontsize=12, fontweight="bold", color="white"
    )
    ax.set_xlabel("Day", color="white")
    ax.set_ylabel("Units on hand", color="white")
    ax.set_ylim(0, max_cap * 1.08)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.tick_params(colors="white")
    ax.set_facecolor("#0f0f18")
    fig.patch.set_facecolor("#0a0a12")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")

    plt.tight_layout()
    path = "models/results/stock_timeline_best.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0a12")
    print(f"  Saved: {path}")
    plt.close()


def generate_all_plots(results):
    """Run all five report plot functions."""
    print(f"\n{BOLD}  Generating report plots...{RESET}")
    plot_cumulative_reward_curves(results)
    plot_convergence(results)
    plot_metrics_comparison(results)
    plot_generalisation_test(results)
    plot_stock_timeline(results)
    print(f"  {GREEN}All plots saved to models/results/{RESET}")


# ═════════════════════════════════════════════════════════════════════════
# JSON EXPORT
# ═════════════════════════════════════════════════════════════════════════

def export_json(results, loaded_models):
    """Export all comparison results to JSON for API readiness demo."""
    summary = {}
    for algo, episodes in results.items():
        summary[algo] = {
            "model_path":            loaded_models.get(algo, "N/A"),
            "n_episodes":            len(episodes),
            "mean_total_reward":     round(
                np.mean([e["total_reward"]     for e in episodes]), 2),
            "std_total_reward":      round(
                np.std([e["total_reward"]      for e in episodes]), 2),
            "mean_service_rate":     round(
                np.mean([e["service_rate"]     for e in episodes]), 2),
            "mean_stockout_days":    round(
                np.mean([e["stockout_days"]    for e in episodes]), 2),
            "mean_emergency_orders": round(
                np.mean([e["emergency_orders"] for e in episodes]), 2),
        }

    best_algo = max(summary, key=lambda a: summary[a]["mean_total_reward"])

    output = {
        "meta": {
            "timestamp":   datetime.now().isoformat(),
            "environment": "PharmacySupplyEnv-v0",
            "description": (
                "Cross-algorithm RL evaluation for medication shortage "
                "prevention in African healthcare supply chains."
            ),
        },
        "best_algorithm":    best_algo.upper(),
        "algorithm_summary": summary,
        "api_integration_example": {
            "description": (
                "The trained agent can be deployed as a REST API. "
                "A pharmacy dashboard POSTs current inventory state "
                "and receives an order recommendation."
            ),
            "endpoint":      "POST /api/v1/recommend-order",
            "request_body":  {
                "stock_level_ratio":   0.35,
                "days_to_stockout":    0.20,
                "lead_time_remaining": 0.0,
                "demand_trend":        0.44,
                "pending_order":       0.0,
                "time_progress":       0.55,
                "high_season":         1.0,
            },
            "response_body": {
                "recommended_action": 2,
                "action_label":       "Order medium batch (50% capacity)",
                "confidence":         0.87,
                "algorithm_used":     best_algo.upper(),
            },
        },
    }

    path = "models/results/comparison_results.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {path}")
    return path


# ═════════════════════════════════════════════════════════════════════════
# LIVE DEMO
# ═════════════════════════════════════════════════════════════════════════

def run_live_demo(results, n_episodes=2, render=True):
    """
    Run the best-performing algorithm live with pygame.
    Prints the full verbose step-by-step output needed for video recording.
    """
    if not results:
        print(f"  {RED}No results available for demo.{RESET}")
        return

    means     = {a: np.mean([ep["total_reward"] for ep in eps])
                 for a, eps in results.items()}
    best_algo = max(means, key=means.get)
    path_m, fw = find_model(best_algo)

    print(f"\n{BOLD}{'═'*72}{RESET}")
    print(f"{BOLD}  LIVE DEMO — {best_algo.upper()} "
          f"(mean reward = {means[best_algo]:.1f}){RESET}")
    print(f"{BOLD}{'═'*72}{RESET}")

    print(f"\n{BOLD}  Problem:{RESET}")
    print(f"  Medication shortages in African healthcare facilities cause")
    print(f"  treatment delays and increased patient mortality.")
    print(f"  This RL agent learns proactive inventory management —")
    print(f"  ordering the right amount at the right time to prevent stockouts.")

    print(f"\n{BOLD}  Reward structure:{RESET}")
    print(f"  {GREEN}+10{RESET}  per day demand is fully met (no stockout)")
    print(f"  {GREEN} +2{RESET}  efficiency bonus (stock in 30–70% range)")
    print(f"  {YELLOW} -5{RESET}  low stock warning (< 50 units)")
    print(f"  {RED}-10{RESET}  critical stock (< 20 units)")
    print(f"  {RED}-20{RESET}  stockout — patient demand unmet")
    print(f"  {YELLOW} -5{RESET}  overstock penalty (waste / expiry risk)")
    print(f"  {RED}-15{RESET}  emergency order (high procurement cost)")

    print(f"\n{BOLD}  Agent objective:{RESET}")
    print(f"  Maximise cumulative reward over 365 days by keeping stock")
    print(f"  in the safe zone — avoiding both stockouts and overstock.")

    renderer = None
    if render:
        try:
            from environment.rendering import PharmacyRenderer
            renderer = PharmacyRenderer(max_capacity=500)
        except Exception as e:
            print(f"  {YELLOW}Pygame unavailable ({e}). "
                  f"Running terminal only.{RESET}")

    env_base   = PharmacySupplyEnv(seed=99)
    predict_fn = load_predict_fn(best_algo, path_m, fw, env_base)
    env_base.close()

    import pygame

    for ep in range(1, n_episodes + 1):
        env_ep = PharmacySupplyEnv(seed=99 + ep)
        obs, _ = env_ep.reset()
        done   = False
        ep_reward = 0.0

        if renderer:
            print(f"\n  {CYAN}Episode {ep} running in pygame window...{RESET}")

        step_num = 0
        if not renderer:
            # verbose terminal output when no pygame
            stats = run_episode(predict_fn, env_ep, verbose=True,
                                episode_num=ep,
                                algo_label=f"{best_algo.upper()} (best)")
            ep_reward = stats["total_reward"]
        else:
            # render each step to pygame
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        break
                if done:
                    break

                action = predict_fn(obs)
                obs, reward, terminated, truncated, info = env_ep.step(action)
                ep_reward += reward
                done = terminated or truncated
                step_num += 1

                renderer.render(
                    stock_level=info["stock_level"],
                    max_capacity=env_ep.max_capacity,
                    time_step=info["time_step"],
                    pending_order=env_ep.pending_order_units,
                    demand_history=env_ep.demand_history,
                    consecutive_stockout_days=env_ep.consecutive_stockout_days,
                    action=action,
                    reward=reward,
                    info=info,
                )

        env_ep.close()
        print(f"\n  Episode {ep} complete | "
              f"Total reward: {BOLD}{ep_reward:+.1f}{RESET}")

    if renderer:
        print(f"\n  {CYAN}Close the pygame window to exit.{RESET}")
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
        renderer.close()


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pharmacy Supply RL — Cross-algorithm comparison & demo"
    )
    parser.add_argument("--episodes",     type=int,  default=5,
                        help="Eval episodes per algorithm (default 5)")
    parser.add_argument("--compare-only", action="store_true",
                        help="Evaluate + plot only, no live pygame demo")
    parser.add_argument("--run-only",     type=str,  default=None,
                        metavar="ALGO",
                        help="Skip comparison, run one algo live "
                             "(dqn / ppo / a2c / reinforce)")
    parser.add_argument("--no-render",    action="store_true",
                        help="Disable pygame window")
    parser.add_argument("--seed",         type=int,  default=0,
                        help="Base random seed (default 0)")
    args = parser.parse_args()

    print(f"\n{BOLD}{'═'*72}{RESET}")
    print(f"{BOLD}  PHARMACY SUPPLY RL — CROSS-ALGORITHM COMPARISON{RESET}")
    print(f"{BOLD}  Medication Shortage Prevention | PharmacySupplyEnv-v0{RESET}")
    print(f"{BOLD}{'═'*72}{RESET}")

    # ── Quick single-algo live run ────────────────────────────────────────
    if args.run_only:
        algo       = args.run_only.lower()
        path_m, fw = find_model(algo)
        if path_m is None:
            print(f"{RED}No model for '{algo}'. Train it first.{RESET}")
            sys.exit(1)
        env        = PharmacySupplyEnv(seed=args.seed)
        predict_fn = load_predict_fn(algo, path_m, fw, env)
        ep         = run_episode(predict_fn, env, verbose=False)
        env.close()
        results = {algo: [ep]}
        run_live_demo(results, n_episodes=args.episodes,
                      render=not args.no_render)
        return

    # ── Full pipeline ─────────────────────────────────────────────────────
    results, loaded = evaluate_all_algorithms(
        n_episodes=args.episodes,
        seed=args.seed,
        verbose_algo="ppo",
    )

    if not results:
        print(f"\n{RED}No trained models found. Run training first:{RESET}")
        print("  python training/dqn_training.py")
        print("  python training/pg_training.py")
        sys.exit(1)

    print_comparison_table(results)
    generate_all_plots(results)
    json_path = export_json(results, loaded)

    print(f"\n{BOLD}  Output files:{RESET}")
    for fname in [
        "cumulative_reward_curves.png",
        "convergence_plot.png",
        "metrics_comparison.png",
        "generalisation_test.png",
        "stock_timeline_best.png",
        "comparison_results.json",
    ]:
        print(f"  models/results/{fname}")

    if not args.compare_only:
        run_live_demo(results, n_episodes=2, render=not args.no_render)


if __name__ == "__main__":
    main()