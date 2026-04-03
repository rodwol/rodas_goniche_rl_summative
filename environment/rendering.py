"""
rendering.py
============
PharmacyRenderer — A pygame-based visual dashboard for the
PharmacySupplyEnv-v0 environment.

Displays in real time:
  - Stock level bar (colour changes: green → yellow → red as stock drops)
  - Daily demand history graph (last 30 days)
  - Cumulative reward tracker
  - Pending order status
  - Season indicator (normal / high-demand season)
  - Day counter and episode stats

Run this file directly to see a random agent demo:
    python environment/rendering.py
"""

import pygame
import numpy as np
import sys
import os

# ── Colour palette ────────────────────────────────────────────────────────
BLACK       = (15,  15,  20)
WHITE       = (240, 240, 240)
DARK_GREY   = (40,  42,  54)
MID_GREY    = (68,  71,  90)
LIGHT_GREY  = (120, 124, 148)
GREEN       = (80,  200, 120)
YELLOW      = (240, 200, 60)
RED         = (220, 80,  80)
ORANGE      = (240, 140, 60)
BLUE        = (80,  160, 240)
PURPLE      = (160, 100, 240)
TEAL        = (60,  200, 180)
WHITE_PANEL = (28,  30,  40)


class PharmacyRenderer:
    """
    Pygame-based renderer for the PharmacySupplyEnv.

    Creates a 900×600 dashboard window showing all key environment
    metrics in real time as the agent takes actions.
    """

    WINDOW_WIDTH  = 900
    WINDOW_HEIGHT = 600
    FPS           = 10          # frames per second (increase to speed up viz)

    def __init__(self, max_capacity=500):
        """
        Initialise pygame and create the display window.

        Args:
            max_capacity: maximum stock units (from the environment)
        """
        pygame.init()
        pygame.display.set_caption("Pharmacy Supply RL — Medication Shortage Prevention")

        self.screen = pygame.display.set_mode(
            (self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        )
        self.clock  = pygame.time.Clock()
        self.max_capacity = max_capacity

        # Font sizes
        self.font_large  = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 16)
        self.font_small  = pygame.font.SysFont("Arial", 13)

        # History buffers for graphs (last 60 days)
        self.stock_history   = []
        self.demand_history  = []
        self.reward_history  = []
        self.cumulative_rewards = []
        self._cumulative     = 0.0

        # Track events to display
        self.last_action_name = "—"
        self.total_stockouts  = 0
        self.total_days       = 0

        # Action labels for display
        self.action_labels = {
            0: "Do nothing",
            1: "Order small (25%)",
            2: "Order medium (50%)",
            3: "Order large (100%)",
            4: "EMERGENCY order",
        }

    # ─────────────────────────────────────────────────────────────────────
    # MAIN RENDER CALL
    # ─────────────────────────────────────────────────────────────────────
    def render(self, stock_level, max_capacity, time_step,
               pending_order, demand_history, consecutive_stockout_days,
               action=None, reward=0.0, info=None):
        """
        Draw one frame of the dashboard.

        Called once per environment step from custom_env.py's render().

        Args:
            stock_level               : current units on hand
            max_capacity              : max units pharmacy can hold
            time_step                 : current day (0–364)
            pending_order             : units currently in transit
            demand_history            : list of recent daily demand values
            consecutive_stockout_days : days in a row with no stock
            action                    : last action taken by the agent (int)
            reward                    : reward received this step
            info                      : optional info dict from env.step()
        """
        # ── Handle pygame quit events ─────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()

        # ── Update internal history ───────────────────────────────────────
        self.stock_history.append(stock_level)
        if len(self.stock_history) > 60:
            self.stock_history.pop(0)

        avg_demand = np.mean(demand_history) if demand_history else 20.0
        self.demand_history.append(avg_demand)
        if len(self.demand_history) > 60:
            self.demand_history.pop(0)

        self._cumulative += reward
        self.cumulative_rewards.append(self._cumulative)
        if len(self.cumulative_rewards) > 60:
            self.cumulative_rewards.pop(0)

        if action is not None:
            self.last_action_name = self.action_labels.get(action, "—")

        if info and info.get("stockout", False):
            self.total_stockouts += 1
        self.total_days = time_step

        # ── Fill background ───────────────────────────────────────────────
        self.screen.fill(BLACK)

        # ── Draw all panels ───────────────────────────────────────────────
        self._draw_header(time_step, consecutive_stockout_days)
        self._draw_stock_bar(stock_level, max_capacity, pending_order)
        self._draw_stock_graph()
        self._draw_demand_graph()
        self._draw_reward_graph()
        self._draw_stats_panel(reward, pending_order, consecutive_stockout_days)

        # ── Update display ────────────────────────────────────────────────
        pygame.display.flip()
        self.clock.tick(self.FPS)

    # ─────────────────────────────────────────────────────────────────────
    # PANEL DRAWING METHODS
    # ─────────────────────────────────────────────────────────────────────
    def _draw_header(self, time_step, consecutive_stockout_days):
        """Top header bar with title, day counter, and season indicator."""
        # Header background
        pygame.draw.rect(self.screen, DARK_GREY, (0, 0, self.WINDOW_WIDTH, 50))

        # Title
        title = self.font_large.render(
            "Pharmacy Supply RL — Medication Shortage Prevention", True, WHITE
        )
        self.screen.blit(title, (15, 14))

        # Day counter
        day_text = self.font_medium.render(f"Day: {time_step}/365", True, TEAL)
        self.screen.blit(day_text, (700, 10))

        # Season indicator
        if 200 <= time_step <= 280:
            season_color = RED
            season_text  = "HIGH SEASON"
        else:
            season_color = GREEN
            season_text  = "Normal season"
        season_surf = self.font_medium.render(season_text, True, season_color)
        self.screen.blit(season_surf, (700, 28))

    def _draw_stock_bar(self, stock_level, max_capacity, pending_order):
        """Large stock level bar in the top-left panel."""
        panel_x, panel_y = 15, 60
        panel_w, panel_h = 280, 180
        self._draw_panel(panel_x, panel_y, panel_w, panel_h, "Stock Level")

        # Bar background (empty)
        bar_x  = panel_x + 20
        bar_y  = panel_y + 40
        bar_w  = panel_w - 40
        bar_h  = 80
        pygame.draw.rect(self.screen, MID_GREY, (bar_x, bar_y, bar_w, bar_h), border_radius=6)

        # Filled portion — colour based on how full the stock is
        ratio = stock_level / max_capacity if max_capacity > 0 else 0
        fill_w = int(bar_w * ratio)

        if ratio > 0.5:
            bar_colour = GREEN
        elif ratio > 0.25:
            bar_colour = YELLOW
        elif ratio > 0.0:
            bar_colour = ORANGE
        else:
            bar_colour = RED

        if fill_w > 0:
            pygame.draw.rect(
                self.screen, bar_colour,
                (bar_x, bar_y, fill_w, bar_h),
                border_radius=6
            )

        # Overstock warning line at 85%
        overstock_x = bar_x + int(bar_w * 0.85)
        pygame.draw.line(self.screen, RED,
                         (overstock_x, bar_y), (overstock_x, bar_y + bar_h), 2)

        # Stock percentage text
        pct_text = self.font_large.render(f"{ratio*100:.1f}%", True, WHITE)
        self.screen.blit(pct_text, (bar_x + bar_w//2 - 30, bar_y + 28))

        # Units text
        units_text = self.font_small.render(
            f"{int(stock_level)} / {max_capacity} units", True, LIGHT_GREY
        )
        self.screen.blit(units_text, (bar_x, bar_y + bar_h + 8))

        # Pending order info
        if pending_order > 0:
            pending_surf = self.font_small.render(
                f"In transit: {int(pending_order)} units", True, BLUE
            )
        else:
            pending_surf = self.font_small.render("No pending order", True, LIGHT_GREY)
        self.screen.blit(pending_surf, (bar_x, bar_y + bar_h + 26))

        # Last action
        action_surf = self.font_small.render(
            f"Last action: {self.last_action_name}", True, PURPLE
        )
        self.screen.blit(action_surf, (bar_x, bar_y + bar_h + 44))

    def _draw_stock_graph(self):
        """Line graph: stock level over last 60 days (top middle)."""
        panel_x, panel_y = 310, 60
        panel_w, panel_h = 280, 180
        self._draw_panel(panel_x, panel_y, panel_w, panel_h, "Stock History (60 days)")

        if len(self.stock_history) < 2:
            return

        graph_x = panel_x + 15
        graph_y = panel_y + 35
        graph_w = panel_w - 30
        graph_h = panel_h - 50

        max_val = self.max_capacity
        self._draw_line_graph(
            self.stock_history, graph_x, graph_y, graph_w, graph_h,
            max_val, GREEN, zero_line=True
        )

        # Danger threshold line at 25% capacity
        thresh_y = graph_y + graph_h - int(graph_h * 0.25)
        pygame.draw.line(self.screen, RED,
                         (graph_x, thresh_y), (graph_x + graph_w, thresh_y), 1)
        thresh_label = self.font_small.render("danger", True, RED)
        self.screen.blit(thresh_label, (graph_x + graph_w - 45, thresh_y - 14))

    def _draw_demand_graph(self):
        """Line graph: average daily demand over last 60 days (top right)."""
        panel_x, panel_y = 605, 60
        panel_w, panel_h = 280, 180
        self._draw_panel(panel_x, panel_y, panel_w, panel_h, "Demand Trend (60 days)")

        if len(self.demand_history) < 2:
            return

        graph_x = panel_x + 15
        graph_y = panel_y + 35
        graph_w = panel_w - 30
        graph_h = panel_h - 50

        max_val = max(max(self.demand_history) * 1.2, 50)
        self._draw_line_graph(
            self.demand_history, graph_x, graph_y, graph_w, graph_h,
            max_val, YELLOW
        )

    def _draw_reward_graph(self):
        """Line graph: cumulative reward over last 60 steps (bottom left)."""
        panel_x, panel_y = 15, 255
        panel_w, panel_h = 280, 200
        self._draw_panel(panel_x, panel_y, panel_w, panel_h, "Cumulative Reward")

        if len(self.cumulative_rewards) < 2:
            return

        graph_x = panel_x + 15
        graph_y = panel_y + 35
        graph_w = panel_w - 30
        graph_h = panel_h - 50

        vals    = self.cumulative_rewards
        min_val = min(vals)
        max_val = max(vals)
        span    = max_val - min_val if max_val != min_val else 1

        # Zero line
        zero_y = graph_y + graph_h - int(graph_h * (0 - min_val) / span)
        zero_y = max(graph_y, min(graph_y + graph_h, zero_y))
        pygame.draw.line(self.screen, LIGHT_GREY,
                         (graph_x, zero_y), (graph_x + graph_w, zero_y), 1)

        # Reward line — green if positive trend, red if negative
        colour = GREEN if vals[-1] >= 0 else RED
        points = []
        for i, v in enumerate(vals):
            x = graph_x + int(i * graph_w / max(len(vals) - 1, 1))
            y = graph_y + graph_h - int((v - min_val) / span * graph_h)
            y = max(graph_y, min(graph_y + graph_h, y))
            points.append((x, y))

        if len(points) >= 2:
            pygame.draw.lines(self.screen, colour, False, points, 2)

        # Current value label
        val_text = self.font_small.render(
            f"Total: {self.cumulative_rewards[-1]:.0f}", True, colour
        )
        self.screen.blit(val_text, (graph_x, graph_y + graph_h + 5))

    def _draw_stats_panel(self, reward, pending_order, consecutive_stockout_days):
        """Stats summary panel — bottom right area."""
        panel_x, panel_y = 310, 255
        panel_w, panel_h = 575, 200
        self._draw_panel(panel_x, panel_y, panel_w, panel_h, "Episode Statistics")

        col1_x = panel_x + 20
        col2_x = panel_x + 310
        row_h  = 28
        start_y = panel_y + 45

        stats_left = [
            ("Total days elapsed",   f"{self.total_days}"),
            ("Total stockout events", f"{self.total_stockouts}"),
            ("Consecutive stockout days", f"{consecutive_stockout_days}"),
            ("Last step reward",     f"{reward:+.1f}"),
            ("Cumulative reward",    f"{self._cumulative:.0f}"),
        ]

        stats_right = [
            ("Current stock",     f"{self.stock_history[-1]:.0f} units"
                                   if self.stock_history else "—"),
            ("Avg daily demand",  f"{self.demand_history[-1]:.1f} units/day"
                                   if self.demand_history else "—"),
            ("Pending order",     f"{int(pending_order)} units"
                                   if pending_order > 0 else "None"),
            ("Stock ratio",       f"{(self.stock_history[-1]/self.max_capacity)*100:.1f}%"
                                   if self.stock_history else "—"),
            ("Last action",       self.last_action_name),
        ]

        for i, (label, value) in enumerate(stats_left):
            label_surf = self.font_small.render(label + ":", True, LIGHT_GREY)
            value_surf = self.font_medium.render(value, True, WHITE)
            self.screen.blit(label_surf, (col1_x, start_y + i * row_h))
            self.screen.blit(value_surf, (col1_x + 195, start_y + i * row_h - 2))

        for i, (label, value) in enumerate(stats_right):
            label_surf = self.font_small.render(label + ":", True, LIGHT_GREY)
            value_surf = self.font_medium.render(value, True, TEAL)
            self.screen.blit(label_surf, (col2_x, start_y + i * row_h))
            self.screen.blit(value_surf, (col2_x + 130, start_y + i * row_h - 2))

        # Bottom legend for stock bar colours
        legend_y = panel_y + panel_h - 28
        for colour, label, lx in [
            (GREEN,  "> 50%: Safe",       panel_x + 20),
            (YELLOW, "25–50%: Low",       panel_x + 145),
            (ORANGE, "1–25%: Critical",   panel_x + 265),
            (RED,    "0%: STOCKOUT",      panel_x + 400),
        ]:
            pygame.draw.rect(self.screen, colour, (lx, legend_y + 4, 12, 12), border_radius=2)
            surf = self.font_small.render(label, True, LIGHT_GREY)
            self.screen.blit(surf, (lx + 16, legend_y + 2))

    # ─────────────────────────────────────────────────────────────────────
    # UTILITY METHODS
    # ─────────────────────────────────────────────────────────────────────
    def _draw_panel(self, x, y, w, h, title):
        """Draw a dark rounded panel with a title."""
        pygame.draw.rect(self.screen, WHITE_PANEL, (x, y, w, h), border_radius=8)
        pygame.draw.rect(self.screen, MID_GREY,    (x, y, w, h), width=1, border_radius=8)
        title_surf = self.font_small.render(title, True, LIGHT_GREY)
        self.screen.blit(title_surf, (x + 10, y + 8))

    def _draw_line_graph(self, data, x, y, w, h, max_val, colour, zero_line=False):
        """Draw a simple line graph from a list of values."""
        if len(data) < 2:
            return

        min_val = 0
        span = max_val - min_val if max_val != min_val else 1

        # Shaded area under the line
        fill_points = [(x, y + h)]
        for i, v in enumerate(data):
            px = x + int(i * w / max(len(data) - 1, 1))
            py = y + h - int((v - min_val) / span * h)
            py = max(y, min(y + h, py))
            fill_points.append((px, py))
        fill_points.append((x + w, y + h))

        fill_colour = tuple(max(0, c - 150) for c in colour)
        if len(fill_points) >= 3:
            pygame.draw.polygon(self.screen, fill_colour, fill_points)

        # Main line
        line_points = []
        for i, v in enumerate(data):
            px = x + int(i * w / max(len(data) - 1, 1))
            py = y + h - int((v - min_val) / span * h)
            py = max(y, min(y + h, py))
            line_points.append((px, py))

        if len(line_points) >= 2:
            pygame.draw.lines(self.screen, colour, False, line_points, 2)

    def close(self):
        """Shut down pygame cleanly."""
        pygame.quit()


# ─────────────────────────────────────────────────────────────────────────
# RANDOM AGENT DEMO
# Run: python environment/rendering.py
# Shows the environment with a random agent — no model, no training.
# This satisfies the assignment requirement for a static random agent demo.
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Add project root to path so imports work
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from environment.custom_env import PharmacySupplyEnv

    print("Starting random agent demo...")
    print("Close the pygame window to exit.\n")

    env      = PharmacySupplyEnv(render_mode=None, seed=42)
    renderer = PharmacyRenderer(max_capacity=env.max_capacity)

    obs, info = env.reset()
    total_reward = 0.0
    step_count   = 0

    running = True
    while running:
        # Check for quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Random action — agent has NO model, just picks randomly
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count   += 1

        # Render one frame
        renderer.render(
            stock_level=info["stock_level"],
            max_capacity=env.max_capacity,
            time_step=info["time_step"],
            pending_order=env.pending_order_units,
            demand_history=env.demand_history,
            consecutive_stockout_days=env.consecutive_stockout_days,
            action=action,
            reward=reward,
            info=info
        )

        if terminated or truncated:
            print(f"Episode ended at day {step_count}.")
            print(f"Total reward: {total_reward:.1f}")
            # Reset and start a new episode automatically
            obs, info    = env.reset()
            total_reward = 0.0
            step_count   = 0

    renderer.close()
    env.close()
    print("Demo closed.")
    