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
    OBSERVATION SPACE (what the agent "sees" each day):
        0 - stock_level         : current units on hand (normalised 0-1)
        1 - days_to_stockout    : estimated days until stock runs out (norm.)
        2 - supplier_lead_time  : days until next order arrives (norm.)
        3 - demand_trend        : rolling 7-day average daily usage (norm.)
        4 - pending_order       : units currently in transit (norm.)
        5 - time_step           : how far through the year we are (norm.)
        6 - season_flag         : 1.0 if high-demand season (e.g. malaria)

    ACTION SPACE (what the agent can do each day):
        0 - Do nothing          : hold current stock, no new order
        1 - Order small batch   : restock 25% of max capacity
        2 - Order medium batch  : restock 50% of max capacity
        3 - Order large batch   : restock 100% of max capacity
        4 - Emergency order     : immediate delivery, very high cost

    REWARD (feedback signal guiding the agent):
        +10  for each day demand is met without a stockout
        +2   bonus for efficient ordering (not over or under stocked)
        -20  penalty when a stockout occurs (patient demand unmet)
        -5   penalty per day of excessive overstock (waste / holding cost)
        -15  penalty for placing an emergency order (high procurement cost)
    -----------------------------------------------------------------------
    """

    # Required by Gymnasium — tells render() what modes are supported
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, max_capacity=500, seed=None):
        """
        Initialise the environment.

        Args:
            render_mode : "human" opens the pygame window, None = no display
            max_capacity: maximum units the pharmacy can hold (default 500)
            seed        : random seed for reproducibility
        """
        super().__init__()

        # ── Environment constants ─────────────────────────────────────────
        self.max_capacity = max_capacity        # max stock the pharmacy holds
        self.episode_length = 365               # one simulated year
        self.overstock_threshold = 0.85         # above 85% capacity = overstock

        # Supplier lead times (days): normal orders take 3-7 days
        self.min_lead_time = 3
        self.max_lead_time = 7
        self.emergency_lead_time = 1            # emergency arrives next day

        # Order sizes as fraction of max_capacity
        self.order_sizes = {
            0: 0,      # do nothing
            1: 0.25,   # small  = 25% of capacity = 125 units
            2: 0.50,   # medium = 50% of capacity = 250 units
            3: 1.00,   # large  = 100% of capacity = 500 units
            4: 0.30,   # emergency = 30% capacity, arrives tomorrow
        }

        # ── Action Space ──────────────────────────────────────────────────
        # Discrete(5) means 5 possible actions: 0, 1, 2, 3, 4
        self.action_space = spaces.Discrete(5)

        # ── Observation Space ─────────────────────────────────────────────
        # Box space: 7 continuous values, all normalised between 0.0 and 1.0
        # This keeps all inputs on the same scale, which helps ML models train
        self.observation_space = spaces.Box(
            low=np.zeros(7, dtype=np.float32),
            high=np.ones(7, dtype=np.float32),
            dtype=np.float32
        )

        # ── Internal state variables (set properly in reset()) ────────────
        self.stock_level = 0
        self.time_step = 0
        self.pending_order_units = 0
        self.pending_order_arrival = -1         # day the order will arrive
        self.consecutive_stockout_days = 0
        self.demand_history = []                # last 7 days of demand
        self.render_mode = render_mode

        # Pygame renderer (only created if render_mode="human")
        self.renderer = None

        # Random number generator — seeded for reproducibility
        self.np_random = np.random.default_rng(seed)

    # ─────────────────────────────────────────────────────────────────────
    # RESET — called at the start of every new episode
    # ─────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        Called once at the start of each episode (each training run).

        The agent starts with a partially stocked pharmacy — not full,
        not empty — to reflect realistic starting conditions.

        Returns:
            observation : the first observation the agent receives
            info        : optional dictionary with extra diagnostic info
        """
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Start with stock between 40% and 80% of capacity (realistic)
        self.stock_level = float(
            self.np_random.integers(
                int(0.4 * self.max_capacity),
                int(0.8 * self.max_capacity)
            )
        )

        self.time_step = 0
        self.pending_order_units = 0
        self.pending_order_arrival = -1
        self.consecutive_stockout_days = 0

        # Initialise demand history with 7 days of average demand
        base_demand = self._get_daily_demand(day=0)
        self.demand_history = [base_demand] * 7

        observation = self._get_observation()
        info = {"stock_level": self.stock_level, "time_step": self.time_step}

        return observation, info

    # ─────────────────────────────────────────────────────────────────────
    # STEP — the core of the environment
    # ─────────────────────────────────────────────────────────────────────
    def step(self, action):
        """
        Execute one day in the simulation.

        The agent picks an action → we simulate what happens that day:
          1. Process any pending order that is due to arrive today
          2. Calculate today's patient demand (with randomness + seasonality)
          3. Fulfill demand from current stock
          4. Calculate the reward based on what happened
          5. Place the new order (if agent chose to order)
          6. Check if the episode should end (terminal condition)
          7. Return the new observation

        Args:
            action : integer 0-4 chosen by the agent

        Returns:
            observation : new state after this step
            reward      : float — feedback signal for the agent
            terminated  : bool — True if episode ends (e.g. 3-day stockout)
            truncated   : bool — True if episode hit max length (365 days)
            info        : dict with extra diagnostic info for logging
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        reward = 0.0
        info = {}

        # ── 1. Receive pending order if it arrives today ──────────────────
        if (self.pending_order_units > 0 and
                self.pending_order_arrival == self.time_step):
            received = min(
                self.pending_order_units,
                self.max_capacity - self.stock_level   # can't exceed capacity
            )
            self.stock_level += received
            self.pending_order_units = 0
            self.pending_order_arrival = -1
            info["order_received"] = received

        # ── 2. Calculate today's patient demand ───────────────────────────
        demand = self._get_daily_demand(self.time_step)
        self.demand_history.append(demand)
        if len(self.demand_history) > 7:
            self.demand_history.pop(0)

        # ── 3. Fulfill demand from stock ──────────────────────────────────
        fulfilled = min(demand, self.stock_level)
        unmet = demand - fulfilled
        self.stock_level -= fulfilled
        self.stock_level = max(0.0, self.stock_level)   # stock can't go negative

        # ── 4. Calculate reward ───────────────────────────────────────────
        if unmet > 0:
            # Stockout: patient demand could not be met
            self.consecutive_stockout_days += 1
            reward += -20.0
            info["stockout"] = True
            info["unmet_demand"] = unmet
        else:
            # Demand fully met — good outcome
            self.consecutive_stockout_days = 0
            reward += 10.0
            info["stockout"] = False

        # Overstock penalty: holding too much stock wastes money and risks expiry
        stock_ratio = self.stock_level / self.max_capacity
        if stock_ratio > self.overstock_threshold:
            reward += -5.0
            info["overstock"] = True
        else:
            info["overstock"] = False

        # Emergency order penalty
        if action == 4:
            reward += -15.0
            info["emergency_order"] = True

        # Efficiency bonus: stock in the "sweet spot" (30-70% capacity)
        if 0.3 <= stock_ratio <= 0.7:
            reward += 2.0

        # ── 5. Place new order if agent chose to order ────────────────────
        if action > 0 and self.pending_order_units == 0:
            order_fraction = self.order_sizes[action]
            order_units = order_fraction * self.max_capacity

            if action == 4:
                # Emergency order: arrives tomorrow
                lead_time = self.emergency_lead_time
            else:
                # Normal order: random lead time between 3-7 days
                lead_time = int(
                    self.np_random.integers(self.min_lead_time,
                                            self.max_lead_time + 1)
                )

            self.pending_order_units = order_units
            self.pending_order_arrival = self.time_step + lead_time
            info["order_placed"] = order_units
            info["lead_time"] = lead_time

        # ── 6. Advance time and check terminal conditions ─────────────────
        self.time_step += 1

        # Terminal condition 1: stockout lasting 3 or more consecutive days
        terminated = self.consecutive_stockout_days >= 3

        # Terminal condition 2: reached end of year (365 days)
        truncated = self.time_step >= self.episode_length

        # ── 7. Return new observation ─────────────────────────────────────
        observation = self._get_observation()
        info["stock_level"] = self.stock_level
        info["time_step"] = self.time_step
        info["reward"] = reward

        # Render if human mode is on
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    # ─────────────────────────────────────────────────────────────────────
    # HELPER: Build the observation vector
    # ─────────────────────────────────────────────────────────────────────
    def _get_observation(self):
        """
        Build the 7-dimensional observation vector the agent receives.
        All values are normalised to [0, 1] so the neural network trains well.
        """
        # Average daily demand over the last 7 days
        avg_demand = np.mean(self.demand_history) if self.demand_history else 1.0
        avg_demand = max(avg_demand, 0.1)  # avoid division by zero

        # Estimated days until stockout at current usage rate
        days_to_stockout = self.stock_level / avg_demand
        days_to_stockout_norm = min(days_to_stockout / 30.0, 1.0)  # cap at 30 days

        # Lead time of current pending order (0 if no order pending)
        if self.pending_order_arrival > 0:
            remaining_lead = self.pending_order_arrival - self.time_step
            lead_norm = min(remaining_lead / self.max_lead_time, 1.0)
        else:
            lead_norm = 0.0

        # Season flag: days 200-280 = malaria / high-demand season in many
        # African countries. Reflects real-world African healthcare context.
        season_flag = 1.0 if 200 <= self.time_step <= 280 else 0.0

        observation = np.array([
            self.stock_level / self.max_capacity,           # 0: stock level
            days_to_stockout_norm,                          # 1: days to stockout
            lead_norm,                                      # 2: lead time remaining
            min(avg_demand / 50.0, 1.0),                   # 3: demand trend
            self.pending_order_units / self.max_capacity,   # 4: pending order
            self.time_step / self.episode_length,           # 5: time progress
            season_flag,                                    # 6: high-demand season
        ], dtype=np.float32)

        return observation

    # ─────────────────────────────────────────────────────────────────────
    # HELPER: Simulate daily patient demand
    # ─────────────────────────────────────────────────────────────────────
    def _get_daily_demand(self, day):
        """
        Generate realistic stochastic daily demand for a medication.

        Demand is based on:
          - Base demand: ~20 units/day on average
          - Seasonality: higher demand during malaria/disease season (days 200-280)
          - Random noise: day-to-day variation using a normal distribution
          - Occasional demand spikes: rare but significant surge events

        This models the real-world dynamics described in the capstone paper
        where African health facilities face unpredictable seasonal surges.
        """
        # Base demand: average 20 units per day
        base = 20.0

        # Seasonal multiplier: 1.8x demand during high season (days 200-280)
        if 200 <= day <= 280:
            seasonal_multiplier = 1.8
        elif 190 <= day <= 200 or 280 <= day <= 290:
            # Transition period: gradual ramp up/down
            seasonal_multiplier = 1.3
        else:
            seasonal_multiplier = 1.0

        # Random daily variation (±30% noise using normal distribution)
        noise = self.np_random.normal(loc=1.0, scale=0.3)
        noise = max(noise, 0.1)  # demand can't be negative

        # Rare demand spike: 3% chance each day (disease outbreak, emergency)
        spike = 3.0 if self.np_random.random() < 0.03 else 1.0

        demand = base * seasonal_multiplier * noise * spike
        return max(1.0, round(demand))  # minimum 1 unit demand per day

    # ─────────────────────────────────────────────────────────────────────
    # RENDER — display the environment visually
    # ─────────────────────────────────────────────────────────────────────
    def render(self):
        """
        Render the environment. If render_mode is "human", opens a pygame
        window. The actual rendering logic lives in rendering.py.
        """
        if self.render_mode == "human":
            # Import here to avoid pygame loading when not needed
            from environment.rendering import PharmacyRenderer
            if self.renderer is None:
                self.renderer = PharmacyRenderer(self.max_capacity)
            self.renderer.render(
                stock_level=self.stock_level,
                max_capacity=self.max_capacity,
                time_step=self.time_step,
                pending_order=self.pending_order_units,
                demand_history=self.demand_history,
                consecutive_stockout_days=self.consecutive_stockout_days
            )

    def close(self):
        """Clean up the pygame window when done."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


# ─────────────────────────────────────────────────────────────────────────
# QUICK TEST — run this file directly to verify the environment works
# python environment/custom_env.py
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing PharmacySupplyEnv...")
    print("=" * 50)

    env = PharmacySupplyEnv(seed=42)
    obs, info = env.reset()

    print(f"Initial observation (7 values):")
    labels = ["stock_level", "days_to_stockout", "lead_time",
              "demand_trend", "pending_order", "time_step", "season_flag"]
    for label, val in zip(labels, obs):
        print(f"  {label:20s}: {val:.4f}")

    print(f"\nRunning 10 random steps...")
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()      # random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"  Day {i+1:2d} | action={action} | "
              f"stock={info['stock_level']:.0f} | "
              f"reward={reward:+.1f} | "
              f"stockout={info.get('stockout', False)}")
        if terminated or truncated:
            break

    print(f"\nTotal reward over 10 steps: {total_reward:.1f}")
    print("Environment test passed!")
    env.close()