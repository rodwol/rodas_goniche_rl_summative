"""
custom_env.py
=============
PharmacySupplyEnv-v0 — A custom Gymnasium environment that simulates
a pharmacy inventory management problem in an African healthcare setting.

The RL agent plays the role of a pharmacy manager. Each day (time step),
it decides how much medication to order. The goal is to avoid stockouts
(running out of medicine) while not overstocking (wasting money/medicine).

Capstone connection:
    This environment models the decision-making layer of the predictive
    medication shortage system described in the capstone research.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PharmacySupplyEnv(gym.Env):
    """
    A Gymnasium-compatible environment simulating pharmacy inventory
    management for a single essential medication over one year (365 days).

    -----------------------------------------------------------------------
    OBSERVATION SPACE (what the agent sees each day):
        0 - stock_level         : current units on hand (normalised 0-1)
        1 - days_to_stockout    : estimated days until stock runs out (norm.)
        2 - supplier_lead_time  : days until next order arrives (norm.)
        3 - demand_trend        : rolling 7-day average daily usage (norm.)
        4 - pending_order       : units currently in transit (norm.)
        5 - time_step           : how far through the year we are (norm.)
        6 - season_flag         : 1.0 if high-demand season (e.g. malaria)

    ACTION SPACE (what the agent can do each day):
        0 - Do nothing          : hold current stock, no new order
        1 - Order small batch   : restock 25% of max capacity (125 units)
        2 - Order medium batch  : restock 50% of max capacity (250 units)
        3 - Order large batch   : restock 100% of max capacity (500 units)
        4 - Emergency order     : immediate delivery (150 units), high cost

    REWARD STRUCTURE:
        +10  demand fully met with no shortage
        +2   efficiency bonus — stock in 30-70% sweet spot
        -5   low stock warning — stock below 50 units (danger approaching)
        -10  critical stock — stock below 20 units (imminent stockout)
        -20  stockout — patient demand completely unmet
        -5   overstock penalty — stock above 85% capacity (waste/expiry risk)
        -15  emergency order penalty — very high procurement cost

    Why low-stock penalties matter:
        Without them the DQN agent learns to do nothing until it stocks out
        because the future stockout penalty is discounted too heavily.
        The low-stock signal gives immediate feedback that stock is falling
        dangerously low, breaking the do-nothing trap.
    -----------------------------------------------------------------------
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, max_capacity=500, seed=None):
        super().__init__()

        # ── Environment constants ─────────────────────────────────────────
        self.max_capacity          = max_capacity
        self.episode_length        = 365
        self.overstock_threshold   = 0.85
        self.low_stock_threshold   = 50     # units — triggers -5 penalty
        self.critical_threshold    = 20     # units — triggers -10 penalty

        self.min_lead_time         = 3
        self.max_lead_time         = 7
        self.emergency_lead_time   = 1

        self.order_sizes = {
            0: 0.00,
            1: 0.25,
            2: 0.50,
            3: 1.00,
            4: 0.30,
        }

        # ── Spaces ────────────────────────────────────────────────────────
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=np.zeros(7, dtype=np.float32),
            high=np.ones(7, dtype=np.float32),
            dtype=np.float32,
        )

        # ── State ─────────────────────────────────────────────────────────
        self.stock_level               = 0.0
        self.time_step                 = 0
        self.pending_order_units       = 0.0
        self.pending_order_arrival     = -1
        self.consecutive_stockout_days = 0
        self.demand_history            = []
        self.render_mode               = render_mode
        self.renderer                  = None
        self.np_random                 = np.random.default_rng(seed)

    # ─────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.stock_level = float(
            self.np_random.integers(
                int(0.4 * self.max_capacity),
                int(0.8 * self.max_capacity),
            )
        )
        self.time_step                 = 0
        self.pending_order_units       = 0.0
        self.pending_order_arrival     = -1
        self.consecutive_stockout_days = 0

        base_demand         = self._get_daily_demand(day=0)
        self.demand_history = [base_demand] * 7

        obs  = self._get_observation()
        info = {"stock_level": self.stock_level, "time_step": 0}
        return obs, info

    # ─────────────────────────────────────────────────────────────────────
    def step(self, action):
        assert self.action_space.contains(action)

        reward = 0.0
        info   = {}

        # 1. Receive any pending order arriving today
        if (self.pending_order_units > 0
                and self.pending_order_arrival == self.time_step):
            received = min(
                self.pending_order_units,
                self.max_capacity - self.stock_level,
            )
            self.stock_level          += received
            self.pending_order_units   = 0.0
            self.pending_order_arrival = -1
            info["order_received"]     = received

        # 2. Simulate today's demand
        demand = self._get_daily_demand(self.time_step)
        self.demand_history.append(demand)
        if len(self.demand_history) > 7:
            self.demand_history.pop(0)

        # 3. Fulfill demand from current stock
        fulfilled         = min(demand, self.stock_level)
        unmet             = demand - fulfilled
        self.stock_level -= fulfilled
        self.stock_level  = max(0.0, self.stock_level)

        # 4. Calculate reward
        stock_ratio = self.stock_level / self.max_capacity

        if unmet > 0:
            # Stockout event
            self.consecutive_stockout_days += 1
            reward += -20.0
            info["stockout"]    = True
            info["unmet_demand"] = unmet
        else:
            self.consecutive_stockout_days = 0
            reward += 10.0
            info["stockout"] = False

        # Low stock warning penalties — teach the agent to reorder early
        if self.stock_level < self.critical_threshold and unmet == 0:
            reward += -10.0   # critical: < 20 units
        elif self.stock_level < self.low_stock_threshold and unmet == 0:
            reward += -5.0    # low: < 50 units

        # Overstock penalty
        if stock_ratio > self.overstock_threshold:
            reward += -5.0
            info["overstock"] = True
        else:
            info["overstock"] = False

        # Emergency order penalty
        if action == 4:
            reward += -15.0
            info["emergency_order"] = True

        # Efficiency bonus
        if 0.3 <= stock_ratio <= 0.7:
            reward += 2.0

        # 5. Place new order if agent chose to order
        if action > 0 and self.pending_order_units == 0:
            order_units = self.order_sizes[action] * self.max_capacity
            lead_time   = (self.emergency_lead_time if action == 4
                           else int(self.np_random.integers(
                               self.min_lead_time, self.max_lead_time + 1)))
            self.pending_order_units   = order_units
            self.pending_order_arrival = self.time_step + lead_time
            info["order_placed"] = order_units
            info["lead_time"]    = lead_time

        # 6. Advance time and check terminal conditions
        self.time_step += 1
        terminated = self.consecutive_stockout_days >= 3
        truncated  = self.time_step >= self.episode_length

        obs = self._get_observation()
        info["stock_level"] = self.stock_level
        info["time_step"]   = self.time_step
        info["reward"]      = reward

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ─────────────────────────────────────────────────────────────────────
    def _get_observation(self):
        avg_demand = max(
            np.mean(self.demand_history) if self.demand_history else 1.0,
            0.1,
        )
        days_to_stockout = min(self.stock_level / avg_demand / 30.0, 1.0)

        if self.pending_order_arrival > 0:
            remaining = self.pending_order_arrival - self.time_step
            lead_norm = min(remaining / self.max_lead_time, 1.0)
        else:
            lead_norm = 0.0

        season_flag = 1.0 if 200 <= self.time_step <= 280 else 0.0

        return np.array([
            self.stock_level / self.max_capacity,   # 0 stock ratio
            days_to_stockout,                        # 1 days to stockout
            lead_norm,                               # 2 pending lead time
            min(avg_demand / 50.0, 1.0),            # 3 demand trend
            self.pending_order_units / self.max_capacity,  # 4 in transit
            self.time_step / self.episode_length,    # 5 time progress
            season_flag,                             # 6 high season flag
        ], dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────
    def _get_daily_demand(self, day):
        """Stochastic demand: base 20 units/day + seasonal spike + noise."""
        base = 20.0
        if 200 <= day <= 280:
            mult = 1.8
        elif 190 <= day < 200 or 280 < day <= 290:
            mult = 1.3
        else:
            mult = 1.0

        noise = max(self.np_random.normal(1.0, 0.3), 0.1)
        spike = 3.0 if self.np_random.random() < 0.03 else 1.0
        return max(1.0, round(base * mult * noise * spike))

    # ─────────────────────────────────────────────────────────────────────
    def render(self):
        if self.render_mode == "human":
            from environment.rendering import PharmacyRenderer
            if self.renderer is None:
                self.renderer = PharmacyRenderer(self.max_capacity)
            self.renderer.render(
                stock_level=self.stock_level,
                max_capacity=self.max_capacity,
                time_step=self.time_step,
                pending_order=self.pending_order_units,
                demand_history=self.demand_history,
                consecutive_stockout_days=self.consecutive_stockout_days,
            )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


# ─────────────────────────────────────────────────────────────────────────
# Quick test: python environment/custom_env.py
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing PharmacySupplyEnv...")
    env = PharmacySupplyEnv(seed=42)
    obs, info = env.reset()
    print(f"Start stock: {info['stock_level']:.0f} / {env.max_capacity}")

    total = 0.0
    for i in range(30):
        # Force a medium order on day 1 to verify ordering logic
        action = 2 if i == 0 else env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total += reward
        tag = "ORDER" if action > 0 else ""
        print(f"  Day {i+1:2d} | action={action} | "
              f"stock={info['stock_level']:5.0f} | "
              f"reward={reward:+6.1f} | {tag}")
        if term or trunc:
            break

    print(f"\nTotal reward: {total:.1f}")
    env.close()
    