import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# GLOBAL PARAMETERS
# -----------------------------
DAYS = 300
BASE_DEMAND = 100
BASE_CAPACITY = 100
BASE_LEAD_TIME = 30

# -----------------------------
# EVENTS
# -----------------------------
covid_event = {
    "capacity": 0.6,
    "demand_multiplier": 1.8,
    "lead_time": 120,
    "risk": 0.7,
    "duration": 200
}

suez_event = {
    "capacity": 0.8,
    "demand_multiplier": 1.0,
    "lead_time": 90,
    "risk": 0.3,
    "duration": 20
}

# -----------------------------
# RL AGENT
# -----------------------------
actions = ["increase_prod", "decrease_prod", "reroute", "stock_up", "do_nothing"]
Q = defaultdict(float)

def get_Q(state, action):
    return Q[(tuple(state), action)]

def choose_action(state, epsilon=0.1):
    if random.random() < epsilon:
        return random.choice(actions)
    return max(actions, key=lambda a: get_Q(state, a))

def update_Q(state, action, reward, next_state, alpha=0.1, gamma=0.9):
    best_next = max(get_Q(next_state, a) for a in actions)
    Q[(tuple(state), action)] += alpha * (
        reward + gamma * best_next - get_Q(state, action)
    )

# -----------------------------
# SIMULATION FUNCTION
# -----------------------------
def simulate(policy="baseline", event=None):
    inventory = 500
    backlog = 0
    pipeline = []
    history_backlog = []

    for day in range(DAYS):

        if event and day < event["duration"]:
            capacity = BASE_CAPACITY * event["capacity"]
            demand = np.random.poisson(BASE_DEMAND * event["demand_multiplier"])
            lead_time = int(np.random.normal(event["lead_time"], 10))
            risk = event["risk"]
        else:
            capacity = BASE_CAPACITY
            demand = np.random.poisson(BASE_DEMAND)
            lead_time = BASE_LEAD_TIME
            risk = 0.1

        if random.random() < risk:
            capacity *= random.uniform(0.5, 0.8)

        state = [int(inventory/50), int(backlog/50), int(demand/50)]

        # POLICY
        if policy == "rl":
            action = choose_action(state)

        elif policy == "heuristic":
            if backlog > 200:
                action = "increase_prod"
            elif inventory < 100:
                action = "stock_up"
            else:
                action = "do_nothing"

        elif policy == "optimization":
            required = backlog + demand - inventory
            capacity = min(capacity, max(required, 0))
            action = "optimized"

        else:
            action = "do_nothing"

        # APPLY ACTION
        if action == "increase_prod":
            capacity *= 1.3
        elif action == "decrease_prod":
            capacity *= 0.8
        elif action == "reroute":
            lead_time *= 0.7
        elif action == "stock_up":
            inventory += 100

        # PRODUCTION
        produced = capacity
        arrival_day = day + max(1, int(lead_time))
        pipeline.append((arrival_day, produced))

        arrivals = [p for p in pipeline if p[0] == day]
        for a in arrivals:
            inventory += a[1]
        pipeline = [p for p in pipeline if p[0] > day]

        # DEMAND
        total_demand = demand + backlog

        if inventory >= total_demand:
            inventory -= total_demand
            backlog = 0
        else:
            backlog = total_demand - inventory
            inventory = 0

        # REWARD
        reward = -backlog * 2 - inventory * 0.5

        next_state = [int(inventory/50), int(backlog/50), int(demand/50)]

        if policy == "rl":
            update_Q(state, action, reward, next_state)

        history_backlog.append(backlog)

    return history_backlog

# -----------------------------
# METRICS
# -----------------------------
def compute_metrics(backlog_series):
    avg_backlog = np.mean(backlog_series)
    max_backlog = np.max(backlog_series)

    peak_index = np.argmax(backlog_series)
    recovery_time = None

    for i in range(peak_index, len(backlog_series)):
        if backlog_series[i] < 10:
            recovery_time = i - peak_index
            break

    return avg_backlog, max_backlog, recovery_time

# -----------------------------
# TRAIN RL
# -----------------------------
for _ in range(50):
    simulate("rl", covid_event)

# -----------------------------
# RUN SCENARIO
# -----------------------------
def run_scenario(event, name):
    baseline = simulate("baseline", event)
    heuristic = simulate("heuristic", event)
    rl = simulate("rl", event)
    opt = simulate("optimization", event)

    print(f"\n--- {name} SCENARIO ---")
    for label, data in zip(
        ["Baseline", "Heuristic", "RL", "Optimization"],
        [baseline, heuristic, rl, opt]
    ):
        avg_b, max_b, rec = compute_metrics(data)
        print(f"{label}: Avg={avg_b:.2f}, Max={max_b:.2f}, Recovery={rec}")

    return baseline, heuristic, rl, opt

# -----------------------------
# RUN BOTH EVENTS
# -----------------------------
covid_results = run_scenario(covid_event, "COVID")
suez_results = run_scenario(suez_event, "SUEZ")

# -----------------------------
# PLOT (CLEARLY LABELED)
# -----------------------------

method_labels = ["Baseline", "Heuristic", "RL Agent", "Optimization"]
colors = ["blue", "orange", "green", "red"]

# COVID GRAPH
plt.figure(figsize=(10, 5))
for data, label, color in zip(covid_results, method_labels, colors):
    plt.plot(data, label=label, color=color, linewidth=2)
plt.title("Backlog over Time — COVID Scenario", fontsize=16)
plt.xlabel("Day", fontsize=12)
plt.ylabel("Backlog", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.show()

# SUEZ GRAPH
plt.figure(figsize=(10, 5))
for data, label, color in zip(suez_results, method_labels, colors):
    plt.plot(data, label=label, color=color, linewidth=2)
plt.title("Backlog over Time — SUEZ Scenario", fontsize=16)
plt.xlabel("Day", fontsize=12)
plt.ylabel("Backlog", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.show()# -----------------------------
# PLOT (CLEARLY LABELED)
# -----------------------------

method_labels = ["Baseline", "Heuristic", "RL Agent", "Optimization"]
colors = ["blue", "orange", "green", "red"]

# COVID GRAPH
plt.figure(figsize=(10, 5))
for data, label, color in zip(covid_results, method_labels, colors):
    plt.plot(data, label=label, color=color, linewidth=2)
plt.title("Backlog over Time — COVID Scenario", fontsize=16)
plt.xlabel("Day", fontsize=12)
plt.ylabel("Backlog", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.show()

# SUEZ GRAPH
plt.figure(figsize=(10, 5))
for data, label, color in zip(suez_results, method_labels, colors):
    plt.plot(data, label=label, color=color, linewidth=2)
plt.title("Backlog over Time — SUEZ Scenario", fontsize=16)
plt.xlabel("Day", fontsize=12)
plt.ylabel("Backlog", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.show()