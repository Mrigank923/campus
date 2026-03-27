---
title: CampusMarketEnv
emoji: "🏪"
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - simulation
---

# CampusMarketEnv

`CampusMarketEnv` is a multi-agent economic simulation built for reinforcement learning and OpenEnv.

The environment models a campus neighborhood as a grid world where shop agents compete for student demand. Each agent chooses:

- a `business_type`
- a `price`

Student clusters then choose where to buy based on:

- distance
- budget
- price sensitivity
- preference match

The goal for each agent is to maximize reward:

```text
reward = profit - penalties
```

Where penalties reflect market oversaturation and poor demand alignment.

## What This Project Includes

- A configurable multi-agent environment in `server/campus_environment.py`
- OpenEnv-compatible action and observation models in `models.py`
- A FastAPI/OpenEnv server entrypoint in `server/app.py`
- A lightweight Python client in `client.py`

## Business Types

The current environment supports these shop categories:

- `fast_food`
- `cafe`
- `food`
- `stationery`
- `general_store`
- `fruits_and_juice`
- `tea_stall`
- `medical_store`
- `restaurants`

Each business type has its own price range with:

- `min_price`
- `max_price`
- `default_price`

This makes the simulation more realistic than using one global price scale for every shop category.

## How The Simulation Works

At a high level, one episode works like this:

1. `reset()` creates a new grid, student clusters, and shops.
2. Each agent submits an action with `business_type` and `price`.
3. The environment normalizes each action and clips the price to that business type's allowed range.
4. Student clusters evaluate all shops using distance, price, affordability, and preference match.
5. Customers are allocated probabilistically.
6. Revenue, profit, competition effects, and demand mismatch penalties are computed.
7. The environment returns the next state, rewards, done flag, and debug info.

## Core API

### Local Environment API

The local Python environment supports:

- `reset() -> dict`
- `step(action_dict) -> tuple[next_state, rewards, done, info]`
- `state() -> dict`

Example:

```python
from server.campus_environment import CampusMarketEnv

env = CampusMarketEnv(seed=21)
state = env.reset()

actions = {
    "agent_0": {"business_type": "cafe", "price": 6.5},
    "agent_1": {"business_type": "tea_stall", "price": 2.0},
}

next_state, rewards, done, info = env.step(actions)

print(rewards)
print(done)
```

### OpenEnv Action Format

The server-facing action payload uses:

```json
{
  "actions": {
    "agent_0": { "business_type": "cafe", "price": 6.5 },
    "agent_1": { "business_type": "tea_stall", "price": 2.0 }
  }
}
```

## Quick Start

### Option 1: Run The Environment Logic Directly

This is the easiest way to verify that the simulator works.

From the project root:

```bash
python server/campus_environment.py
```

You should see:

- the initial state keys
- five reward dictionaries from five random steps

This direct mode is useful for:

- smoke testing
- RL environment debugging
- experimenting with logic before running the HTTP server

### Option 2: Run The HTTP/OpenEnv Server

Install dependencies first:

```bash
pip install -r server/requirements.txt
```

Then start the server:

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Useful endpoints:

- `http://localhost:8000/docs`
- `http://localhost:8000/health`
- `http://localhost:8000/ws`

## Verifying That It Works

### Smoke Test

Run:

```bash
python server/campus_environment.py
```

Expected result:

- no crash
- valid reward dictionaries for each step

### Determinism Check

Run:

```bash
python - <<'PY'
from server.campus_environment import CampusMarketEnv

env1 = CampusMarketEnv(seed=11)
env2 = CampusMarketEnv(seed=11)

state1 = env1.reset()
state2 = env2.reset()

actions1 = env1.sample_random_actions()
actions2 = env2.sample_random_actions()

step1 = env1.step(actions1)
step2 = env2.step(actions2)

print("same shops:", state1["shops"] == state2["shops"])
print("same actions:", actions1 == actions2)
print("same rewards:", step1[1] == step2[1])
print("same metrics:", step1[0]["latest_step_metrics"] == step2[0]["latest_step_metrics"])
PY
```

Expected result:

- all lines print `True`

### Type-Specific Price Check

Run:

```bash
python - <<'PY'
from server.campus_environment import CampusMarketEnv

env = CampusMarketEnv(seed=4)
env.reset()
_, _, _, info = env.step({
    "agent_0": {"business_type": "tea_stall", "price": 99},
    "agent_1": {"business_type": "restaurants", "price": -5},
})

print(info["normalized_actions"]["agent_0"])
print(info["normalized_actions"]["agent_1"])
PY
```

This confirms that each type uses its own allowed price band.

## State Structure

The environment state is a structured Python dictionary. Key fields include:

- `episode_id`
- `time_step`
- `max_steps`
- `done`
- `business_types`
- `price_ranges_by_type`
- `grid_size`
- `grid_layout`
- `shops`
- `student_clusters`
- `demand_signals`
- `latest_rewards`
- `latest_step_metrics`

Each shop record includes:

- `agent_id`
- `position`
- `business_type`
- `price`
- `price_range`

## Reward Design

Each agent reward is based on:

```text
reward = profit - competition_penalty - demand_mismatch_penalty
```

Profit is driven by:

- number of customers
- chosen price
- operating cost

Penalties are driven by:

- same-type competition nearby
- poor fit between local demand and realized customers

## Customizing The Environment

Most customization happens inside `server/campus_environment.py`.

### 1. Add Or Remove Business Types

Update:

- `BUSINESS_TYPES`
- `_build_default_price_ranges()`

If you add a business type, make sure it has:

- a price range
- student preference support
- valid initialization behavior

### 2. Change Price Bands

Edit `_build_default_price_ranges()` to adjust:

- `min_price`
- `max_price`
- `default_price`

This is the safest place to tune pricing realism.

### 3. Change Grid Size Or Episode Length

When creating the environment:

```python
env = CampusMarketEnv(
    grid_size=(12, 12),
    num_agents=6,
    num_student_clusters=10,
    max_steps=100,
    seed=42,
)
```

### 4. Change Reward Behavior

Tune these config fields in `EnvironmentConfig`:

- `base_operating_cost`
- `variable_cost_ratio`
- `competition_penalty_weight`
- `demand_mismatch_penalty_weight`
- `distance_weight`
- `preference_weight`
- `price_weight`
- `budget_weight`

### 5. Change Demand Stochasticity

Tune:

- `demand_noise_scale`

Lower values make demand more stable.
Higher values make the simulation noisier.

### 6. Change Student Behavior

The main customer-choice logic lives in:

- `_score_shop()`
- `_allocate_customers()`

That is where to customize:

- distance decay
- budget pressure
- price sensitivity
- preference effects
- outside option behavior

## File Guide

```text
campus/
├── README.md
├── models.py
├── client.py
├── openenv.yaml
├── pyproject.toml
└── server/
    ├── app.py
    ├── campus_environment.py
    ├── Dockerfile
    └── __init__.py
```

### Important Files

- `server/campus_environment.py`: simulation logic, configuration, reward model
- `models.py`: OpenEnv action and observation schemas
- `server/app.py`: HTTP/WebSocket server setup
- `client.py`: Python client wrapper

## Deploying

### Build Docker Image

```bash
docker build -t campus-env:latest -f server/Dockerfile .
```

### Push With OpenEnv

From the project root:

```bash
openenv push
```

Or:

```bash
openenv push --namespace my-org --private
```

## Troubleshooting

### `python server/campus_environment.py` fails

Check:

- you are running the command from the project root
- your Python version is 3.10+

### The server does not start

Check that required packages are installed:

```bash
pip install -r server/requirements.txt
```

### Prices look wrong

Inspect:

- `price_ranges_by_type` in returned state
- `info["normalized_actions"]` after each step

This will show which price band was actually applied.

### I want to change the simulation for my own use case

Start with these safe customization points:

- business catalog and price ranges
- reward weights
- demand noise
- student preference generation
- grid size and episode length

## Suggested Next Improvements

If you want to extend this project further, good next steps are:

- business-type-specific operating costs
- richer student segments
- moving shop locations as part of the action space
- inventory constraints
- seasonality or weekday demand shifts
- explicit rent by grid cell

## License

This project follows the repository license terms.
