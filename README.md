# Pharmacy Supply RL — Medication Shortage Prevention

A reinforcement learning system that trains an agent to proactively manage pharmacy inventory and prevent medication stockouts in African healthcare facilities.

---

## Problem Statement
Medication shortages in African healthcare systems lead to treatment delays, increased morbidity, and higher costs. This project trains an RL agent to learn optimal procurement decisions — ordering the right quantity at the right time — to minimise stockouts while controlling costs.

---

## Project Structure
```
project_root/
├── environment/
│   ├── custom_env.py       # PharmacySupplyEnv-v0 (Gymnasium-compatible)
│   └── rendering.py        # Pygame-based visualization GUI
├── training/
│   ├── dqn_training.py     # DQN training script (value-based)
│   └── pg_training.py      # REINFORCE, PPO, A2C training scripts
├── models/
│   ├── dqn/                # Saved DQN model checkpoints
│   └── pg/                 # Saved policy gradient model checkpoints
├── main.py                 # Runs the best-performing trained agent
├── requirements.txt        # All project dependencies
└── README.md               # This file
```

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/rodwol/rodas_goniche_rl_summative.git
cd rodas_goniche_rl_summative
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Project

### Run random agent demo (no training — just visualization)
```bash
python environment/rendering.py
```

### Train all models
```bash
python training/dqn_training.py
python training/pg_training.py
```

### Run best-performing agent
```bash
python main.py
```

---

## Environment: PharmacySupplyEnv-v0

| Component | Description |
|-----------|-------------|
| **Action space** | Discrete(5) — do nothing, small/medium/large order, emergency order |
| **Observation space** | 7 continuous values: stock level, days to stockout, lead time, demand trend, pending orders, time step, season flag |
| **Reward** | +10 stockout avoided, −20 stockout, −5 overstock, −15 emergency order, +2 efficient order |
| **Episode length** | 365 steps (1 simulated year) |
| **Terminal condition** | Stockout lasting 3+ consecutive days OR end of year |

---

## Algorithms Compared

| Algorithm | Type | Library |
|-----------|------|---------|
| DQN | Value-based | Stable Baselines 3 |
| REINFORCE | Policy Gradient | Custom (PyTorch) |
| PPO | Policy Gradient | Stable Baselines 3 |

---
