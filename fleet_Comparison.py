import os
import copy
import random
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import DataGenerationDynamic as dgd
import fleet_greedy_allocationDynamic as ga
import fleet_auction_allocationDynamic as aa
import fleet_qlearning_allocationDynamic as qa

from reproducibility_utils import (
    set_global_seed,
    create_run_directory,
    save_dataframe,
    save_json
)

# =========================================================
# GLOBAL PARAMETERS
# =========================================================
GLOBAL_BASE_SEED = 42
TIME_STEPS = 100
NEW_TASK_PROB = 0.5
URGENCY_INCREMENT = 1

NUM_VEHICLES = 10
NUM_INITIAL_TASKS = 10

# PHASE 1: TRAINING CONFIGURATION
TRAIN_EPISODES = 1000
EPSILON_START = 1.0
EPSILON_END = 0.01
DECAY_HORIZON = 750.0

# PHASE 2: EVALUATION CONFIGURATION
EVAL_RUNS = 1000 

# =========================================================
# SESSION DIRECTORY
# =========================================================
BASE_RESULTS_DIR = "results"
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, f"session_{SESSION_ID}")
os.makedirs(RESULTS_DIR, exist_ok=True)

# METRIC STORAGE FOR EVALUATION PHASE
eval_results = {
    "Greedy": [],
    "Auction": [],
    "Q-Learning": []
}

# =========================================================
# METRIC FUNCTION
# =========================================================
def compute_metrics(engagement_metrics, vehicles_df, remaining_tasks):
    if not engagement_metrics:
        return {
            "total_tasks": remaining_tasks, "tasks_completed": 0, "tasks_not_serviced": remaining_tasks,
            "avg_engagement_time": 0, "avg_normalized_engagement": 0, "energy_per_unit": 0,
            "throughput": 0, "total_idle_time": vehicles_df["Idle Time"].sum(), "avg_idle_time": vehicles_df["Idle Time"].mean()
        }

    total_tasks = len(engagement_metrics) + remaining_tasks
    completed = len(engagement_metrics)
    total_engagement_time = sum(item["engagement_time"] for item in engagement_metrics)
    avg_engagement_time = total_engagement_time / completed
    avg_normalized_engagement = sum(item["normalized_engagement_time"] for item in engagement_metrics) / completed
    total_task_duration = sum(item["task_duration"] for item in engagement_metrics)
    total_energy_consumed = sum(item["energy_consumed"] for item in engagement_metrics)

    return {
        "total_tasks": total_tasks,
        "tasks_completed": completed,
        "tasks_not_serviced": remaining_tasks,
        "avg_engagement_time": avg_engagement_time,
        "avg_normalized_engagement": avg_normalized_engagement,
        "energy_per_unit": total_energy_consumed / total_task_duration if total_task_duration > 0 else 0,
        "throughput": total_task_duration,
        "total_idle_time": vehicles_df["Idle Time"].sum(),
        "avg_idle_time": vehicles_df["Idle Time"].sum() / len(vehicles_df)
    }

# =========================================================
# PHASE 1: AGENT POLICY TRAINING LOOP (Q-LEARNING ONLY)
# =========================================================
print("=================================================")
print(f"STARTING AGENT TRAINING PHASE ({TRAIN_EPISODES} EPISODES)")
print("=================================================")

for episode in range(TRAIN_EPISODES):
    if episode < DECAY_HORIZON:
        current_epsilon = EPSILON_START - ((EPSILON_START - EPSILON_END) * (episode / DECAY_HORIZON))
    else:
        current_epsilon = EPSILON_END

    if (episode + 1) % 100 == 0 or episode == 0:
        print(f"Training Episode {episode + 1}/{TRAIN_EPISODES} | Current Epsilon: {current_epsilon:.4f}")

    train_seed = GLOBAL_BASE_SEED + episode
    set_global_seed(train_seed)

    vehicles_df = dgd.generate_vehicle_data(NUM_VEHICLES)
    vehicles_qlearning = vehicles_df.copy(deep=True)
    # Change that block in your comparison script to:
    for col in ["Busy", "Remaining Duration", "Idle Time"]: 
        vehicles_qlearning[col] = 0.0

    # Explicitly cast to proper data types:
    vehicles_qlearning["Busy"] = vehicles_qlearning["Busy"].astype(bool)
    vehicles_qlearning["Battery Level (%)"] = vehicles_qlearning["Battery Level (%)"].astype(float)

    tasks_waiting_qlearning = dgd.generate_task_data(NUM_INITIAL_TASKS).to_dict("records")
    task_counter = 1

    for t in range(TIME_STEPS):
        for idx, vehicle in vehicles_qlearning.iterrows():
            if vehicle["Busy"]:
                vehicles_qlearning.at[idx, "Remaining Duration"] -= 1
                if vehicles_qlearning.at[idx, "Remaining Duration"] <= 0:
                    vehicles_qlearning.at[idx, "Busy"] = False
                    vehicles_qlearning.at[idx, "Remaining Duration"] = 0
            else:
                vehicles_qlearning.at[idx, "Idle Time"] += 1

        for task in tasks_waiting_qlearning: task["Urgency"] += URGENCY_INCREMENT

        if random.random() < NEW_TASK_PROB:
            new_task = {
                "Task ID": f"T{task_counter}",
                "Task Position (x, y)": (random.randint(0, 100), random.randint(0, 100)),
                "Urgency": random.randint(0, 9), "Duration (min)": random.randint(10, 30)
            }
            task_counter += 1
            tasks_waiting_qlearning.append(new_task)

        if tasks_waiting_qlearning:
            tasks_df = pd.DataFrame(tasks_waiting_qlearning)
            alloc, _ = qa.qlearning_allocation(vehicles_qlearning, tasks_df, epsilon=current_epsilon)
            tasks_waiting_qlearning = [tk for tk in tasks_waiting_qlearning if tk["Task ID"] not in set(alloc.keys())]

print("\n--> Training Complete. Freezing Q-Table Weights (Setting Epsilon = 0.00).\n")

# =========================================================
# PHASE 2: COMPARATIVE BENCHMARKING (1000 EVALUATION RUNS)
# =========================================================
print("=================================================")
print(f"STARTING COMPARATIVE BENCHMARKING PHASE ({EVAL_RUNS} EVAL RUNS)")
print("=================================================")

for run_id in range(EVAL_RUNS):
    if (run_id + 1) % 100 == 0 or run_id == 0:
        print(f"Executing Evaluation Run {run_id + 1}/{EVAL_RUNS}...")

    eval_seed = GLOBAL_BASE_SEED + TRAIN_EPISODES + run_id
    set_global_seed(eval_seed)
    run_dir = create_run_directory(RESULTS_DIR, run_id)

    vehicles_df = dgd.generate_vehicle_data(NUM_VEHICLES)
    initial_tasks = dgd.generate_task_data(NUM_INITIAL_TASKS)

    save_dataframe(vehicles_df, os.path.join(run_dir, "initial_vehicles.csv"))
    save_dataframe(initial_tasks, os.path.join(run_dir, "initial_tasks.csv"))

    vehicles_greedy = vehicles_df.copy(deep=True)
    vehicles_auction = vehicles_df.copy(deep=True)
    vehicles_qlearning = vehicles_df.copy(deep=True)

    for vehicles in [vehicles_greedy, vehicles_auction, vehicles_qlearning]:
        vehicles["Busy"] = False
        vehicles["Remaining Duration"] = 0.0
        vehicles["Idle Time"] = 0.0
        vehicles["Busy"] = vehicles["Busy"].astype(bool) # Cast guarantee
        vehicles["Battery Level (%)"] = vehicles["Battery Level (%)"].astype(float)

    tasks_waiting_greedy = initial_tasks.to_dict("records")
    tasks_waiting_auction = initial_tasks.to_dict("records")
    tasks_waiting_qlearning = initial_tasks.to_dict("records")

    allocations_greedy, allocations_auction, allocations_qlearning = {}, {}, {}
    engagement_metrics_greedy, engagement_metrics_auction, engagement_metrics_qlearning = [], [], []
    dynamic_tasks_log, vehicle_state_log, waiting_task_log = [], [], []
    task_counter = 1

    for t in range(TIME_STEPS):
        for method_name, vehicles in [("Greedy", vehicles_greedy), ("Auction", vehicles_auction), ("Q-Learning", vehicles_qlearning)]:
            for idx, vehicle in vehicles.iterrows():
                if vehicle["Busy"]:
                    vehicles.at[idx, "Remaining Duration"] -= 1
                    if vehicles.at[idx, "Remaining Duration"] <= 0:
                        vehicles.at[idx, "Busy"] = False
                        vehicles.at[idx, "Remaining Duration"] = 0
                else:
                    vehicles.at[idx, "Idle Time"] += 1

                vehicle_state_log.append({
                    "time_step": t, "method": method_name, "vehicle_id": vehicle["Vehicle ID"],
                    "position": vehicle["Vehicle Position (x, y)"], "battery": vehicle["Battery Level (%)"],
                    "busy": vehicle["Busy"], "remaining_duration": vehicles.at[idx, "Remaining Duration"],
                    "idle_time": vehicles.at[idx, "Idle Time"]
                })

        for task in tasks_waiting_greedy: task["Urgency"] += URGENCY_INCREMENT
        for task in tasks_waiting_auction: task["Urgency"] += URGENCY_INCREMENT
        for task in tasks_waiting_qlearning: task["Urgency"] += URGENCY_INCREMENT

        for method_name, task_list in [("Greedy", tasks_waiting_greedy), ("Auction", tasks_waiting_auction), ("Q-Learning", tasks_waiting_qlearning)]:
            for task in task_list:
                waiting_task_log.append({
                    "time_step": t, "method": method_name, "task_id": task["Task ID"],
                    "position": task["Task Position (x, y)"], "urgency": task["Urgency"], "duration": task["Duration (min)"]
                })

        if random.random() < NEW_TASK_PROB:
            new_task = {
                "Task ID": f"E{task_counter}",
                "Task Position (x, y)": (random.randint(0, 100), random.randint(0, 100)),
                "Urgency": random.randint(0, 9), "Duration (min)": random.randint(10, 30)
            }
            dynamic_tasks_log.append({"time_step": t, **new_task})
            task_counter += 1
            tasks_waiting_greedy.append(copy.deepcopy(new_task))
            tasks_waiting_auction.append(copy.deepcopy(new_task))
            tasks_waiting_qlearning.append(copy.deepcopy(new_task))

        if tasks_waiting_greedy:
            alloc, details = ga.greedy_allocation(vehicles_greedy, pd.DataFrame(tasks_waiting_greedy))
            allocations_greedy.update(alloc)
            engagement_metrics_greedy.extend(details)
            tasks_waiting_greedy = [tk for tk in tasks_waiting_greedy if tk["Task ID"] not in set(alloc.keys())]

        if tasks_waiting_auction:
            alloc, details = aa.auction_allocation(vehicles_auction, pd.DataFrame(tasks_waiting_auction))
            allocations_auction.update(alloc)
            engagement_metrics_auction.extend(details)
            tasks_waiting_auction = [tk for tk in tasks_waiting_auction if tk["Task ID"] not in set(alloc.keys())]

        if tasks_waiting_qlearning:
            alloc, details = qa.qlearning_allocation(vehicles_qlearning, pd.DataFrame(tasks_waiting_qlearning), epsilon=0.00)
            allocations_qlearning.update(alloc)
            engagement_metrics_qlearning.extend(details)
            tasks_waiting_qlearning = [tk for tk in tasks_waiting_qlearning if tk["Task ID"] not in set(alloc.keys())]

    # Compute & save files per run
    metrics_greedy = compute_metrics(engagement_metrics_greedy, vehicles_greedy, len(tasks_waiting_greedy))
    metrics_auction = compute_metrics(engagement_metrics_auction, vehicles_auction, len(tasks_waiting_auction))
    metrics_qlearning = compute_metrics(engagement_metrics_qlearning, vehicles_qlearning, len(tasks_waiting_qlearning))

    eval_results["Greedy"].append(metrics_greedy)
    eval_results["Auction"].append(metrics_auction)
    eval_results["Q-Learning"].append(metrics_qlearning)

    save_dataframe(pd.DataFrame(dynamic_tasks_log), os.path.join(run_dir, "dynamic_tasks.csv"))
    save_dataframe(pd.DataFrame(vehicle_state_log), os.path.join(run_dir, "vehicle_states.csv"))
    save_dataframe(pd.DataFrame(waiting_task_log), os.path.join(run_dir, "waiting_tasks.csv"))
    save_json(allocations_greedy, os.path.join(run_dir, "greedy_allocations.json"))
    save_json(allocations_auction, os.path.join(run_dir, "auction_allocations.json"))
    save_json(allocations_qlearning, os.path.join(run_dir, "qlearning_allocations.json"))
    save_dataframe(pd.DataFrame(engagement_metrics_greedy), os.path.join(run_dir, "greedy_engagement_metrics.csv"))
    save_dataframe(pd.DataFrame(engagement_metrics_auction), os.path.join(run_dir, "auction_engagement_metrics.csv"))
    save_dataframe(pd.DataFrame(engagement_metrics_qlearning), os.path.join(run_dir, "qlearning_engagement_metrics.csv"))

# =========================================================
# AGGREGATION & GRAPH GENERATION
# =========================================================
metric_keys = ["total_tasks", "tasks_completed", "tasks_not_serviced", "avg_engagement_time", "avg_normalized_engagement", "energy_per_unit", "throughput", "total_idle_time", "avg_idle_time"]
aggregate_rows = []

for method in eval_results:
    for run_idx, metrics in enumerate(eval_results[method]):
        row = {"run_id": run_idx, "method": method}
        row.update(metrics)
        aggregate_rows.append(row)

aggregate_df = pd.DataFrame(aggregate_rows)
save_dataframe(aggregate_df, os.path.join(RESULTS_DIR, "eval_aggregate_metrics.csv"))

average_results = {method: {key: np.mean([run[key] for run in eval_results[method]]) for key in metric_keys} for method in eval_results}
summary_df = pd.DataFrame(average_results).T
save_dataframe(summary_df, os.path.join(RESULTS_DIR, "eval_aggregate_summary.csv"))

print("\n=================================================")
print("FINAL SUMMARY RESULTS (1,000 FROZEN EVAL RUNS)")
print("=================================================")
print(summary_df)

# BAR CHART GENERATION
methods = ["Greedy", "Auction", "Q-Learning"]
metrics_labels = ["tasks_completed", "tasks_not_serviced", "avg_engagement_time", "avg_normalized_engagement", "throughput", "energy_per_unit", "total_idle_time", "avg_idle_time"]
titles = ["Tasks Completed\n(Higher Better)", "Tasks Not Serviced\n(Lower Better)", "Avg Engagement Time\n(Lower Better)", "Avg Normalized Engagement\n(Lower Better)", "Throughput\n(Higher Better)", "Energy per Unit Task\n(Lower Better)", "Total Idle Time\n(Lower Better)", "Avg Idle Time\n(Lower Better)"]

fig, axes = plt.subplots(2, 4, figsize=(24, 10))
axes = axes.flatten()
for i, metric in enumerate(metrics_labels):
    values = [average_results[m][metric] for m in methods]
    axes[i].bar(methods, values)
    axes[i].set_title(titles[i])
    axes[i].set_ylabel(metric)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "comparison_metrics.png"), dpi=300, bbox_inches="tight")
plt.close()

# HISTOGRAMS (comparison_distributions.png)
fig, axes = plt.subplots(2, 4, figsize=(24, 10))
axes = axes.flatten()
for i, metric in enumerate(metrics_labels):
    for m in methods:
        axes[i].hist([r[metric] for r in eval_results[m]], alpha=0.5, label=m)
    axes[i].set_title(metric)
    axes[i].legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "comparison_distributions.png"), dpi=300)
plt.close()

# TREND LINES (comparison_trends.png)
fig, axes = plt.subplots(2, 4, figsize=(24, 10))
axes = axes.flatten()
for i, metric in enumerate(metrics_labels):
    for m in methods:
        axes[i].plot([r[metric] for r in eval_results[m]], label=m)
    axes[i].set_title(metric)
    axes[i].legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "comparison_trends.png"), dpi=300)
plt.close()

print("\n=================================================")
print(f"ALL FILES SUCCESSFULLY LOGGED AND SAVED TO:\n{RESULTS_DIR}")
print("=================================================")