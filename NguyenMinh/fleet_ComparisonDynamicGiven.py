# fleet_ComparisonDynamicGiven.py

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

NUM_RUNS = 1000

NUM_VEHICLES = 10

NUM_INITIAL_TASKS = 10


# =========================================================
# SESSION DIRECTORY
# =========================================================

BASE_RESULTS_DIR = "results"

SESSION_ID = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

RESULTS_DIR = os.path.join(
    BASE_RESULTS_DIR,
    f"session_{SESSION_ID}"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# =========================================================
# STORAGE
# =========================================================

all_results = {

    "Greedy": [],

    "Auction": [],

    "Q-Learning": []
}


# =========================================================
# METRIC FUNCTION
# =========================================================

def compute_metrics(
    engagement_metrics,
    vehicles_df,
    remaining_tasks
):

    if not engagement_metrics:

        total_tasks = remaining_tasks

        return {

            "total_tasks": total_tasks,

            "tasks_completed": 0,

            "tasks_not_serviced":
                remaining_tasks,

            "avg_engagement_time": 0,

            "avg_normalized_engagement": 0,

            "energy_per_unit": 0,

            "throughput": 0,

            "total_idle_time":
                vehicles_df["Idle Time"].sum(),

            "avg_idle_time":
                vehicles_df["Idle Time"].mean()
        }

    total_tasks = len(engagement_metrics) + remaining_tasks
    completed = len(engagement_metrics)

    total_engagement_time = sum(

        item["engagement_time"]

        for item in engagement_metrics
    )

    avg_engagement_time = (

        total_engagement_time
        / completed
    )

    avg_normalized_engagement = sum(

        item[
            "normalized_engagement_time"
        ]

        for item in engagement_metrics

    ) / completed

    total_task_duration = sum(

        item["task_duration"]

        for item in engagement_metrics
    )

    total_energy_consumed = sum(

        item["energy_consumed"]

        for item in engagement_metrics
    )

    energy_per_unit = (

        total_energy_consumed
        / total_task_duration

        if total_task_duration > 0
        else 0
    )

    throughput = total_task_duration

    total_idle_time = (
        vehicles_df["Idle Time"].sum()
    )

    avg_idle_time = (

        total_idle_time
        / len(vehicles_df)
    )

    return {

        "total_tasks":
            total_tasks,

        "tasks_completed":
            completed,

        "tasks_not_serviced":
            remaining_tasks,

        "avg_engagement_time":
            avg_engagement_time,

        "avg_normalized_engagement":
            avg_normalized_engagement,

        "energy_per_unit":
            energy_per_unit,

        "throughput":
            throughput,

        "total_idle_time":
            total_idle_time,

        "avg_idle_time":
            avg_idle_time
    }


# =========================================================
# RUN EXPERIMENTS
# =========================================================

for run_id in range(NUM_RUNS):

    print("\n=================================================")
    print(f"RUN {run_id + 1}/{NUM_RUNS}")
    print("=================================================")

    run_seed = (
        GLOBAL_BASE_SEED + run_id
    )

    set_global_seed(
        run_seed
    )

    run_dir = create_run_directory(
        RESULTS_DIR,
        run_id
    )

    # =====================================================
    # INITIAL ENVIRONMENT
    # =====================================================

    vehicles_df = dgd.generate_vehicle_data(
        NUM_VEHICLES
    )

    initial_tasks = dgd.generate_task_data(
        NUM_INITIAL_TASKS
    )

    save_dataframe(

        vehicles_df,

        os.path.join(
            run_dir,
            "initial_vehicles.csv"
        )
    )

    save_dataframe(

        initial_tasks,

        os.path.join(
            run_dir,
            "initial_tasks.csv"
        )
    )

    # =====================================================
    # COPY ENVIRONMENTS
    # =====================================================

    vehicles_greedy = (
        vehicles_df.copy(deep=True)
    )

    vehicles_auction = (
        vehicles_df.copy(deep=True)
    )

    vehicles_qlearning = (
        vehicles_df.copy(deep=True)
    )

    # =====================================================
    # ADD TRACKING COLUMNS
    # =====================================================

    for vehicles in [

        vehicles_greedy,

        vehicles_auction,

        vehicles_qlearning
    ]:

        vehicles["Busy"] = False

        vehicles["Remaining Duration"] = 0.0

        vehicles["Idle Time"] = 0.0

        vehicles["Battery Level (%)"] = (

            vehicles["Battery Level (%)"]
            .astype(float)
        )

    # =====================================================
    # TASK LISTS
    # =====================================================

    tasks_waiting_greedy = (
        initial_tasks.to_dict(
            "records"
        )
    )

    tasks_waiting_auction = (
        initial_tasks.to_dict(
            "records"
        )
    )

    tasks_waiting_qlearning = (
        initial_tasks.to_dict(
            "records"
        )
    )

    # =====================================================
    # STORAGE
    # =====================================================

    allocations_greedy = {}

    allocations_auction = {}

    allocations_qlearning = {}

    engagement_metrics_greedy = []

    engagement_metrics_auction = []

    engagement_metrics_qlearning = []

    dynamic_tasks_log = []

    vehicle_state_log = []

    waiting_task_log = []

    task_counter = 1

    # =====================================================
    # SIMULATION LOOP
    # =====================================================

    for t in range(TIME_STEPS):

        # -------------------------------------------------
        # UPDATE VEHICLES
        # -------------------------------------------------

        for method_name, vehicles in [

            ("Greedy", vehicles_greedy),

            ("Auction", vehicles_auction),

            ("Q-Learning", vehicles_qlearning)
        ]:

            for idx, vehicle in vehicles.iterrows():

                if vehicle["Busy"]:

                    new_duration = (

                        vehicle[
                            "Remaining Duration"
                        ] - 1
                    )

                    vehicles.at[
                        idx,
                        "Remaining Duration"
                    ] = new_duration

                    if new_duration <= 0:

                        vehicles.at[
                            idx,
                            "Busy"
                        ] = False

                        vehicles.at[
                            idx,
                            "Remaining Duration"
                        ] = 0

                else:

                    vehicles.at[
                        idx,
                        "Idle Time"
                    ] += 1

                vehicle_state_log.append({

                    "time_step": t,

                    "method": method_name,

                    "vehicle_id":
                        vehicle["Vehicle ID"],

                    "position":
                        vehicle[
                            "Vehicle Position (x, y)"
                        ],

                    "battery":
                        vehicle[
                            "Battery Level (%)"
                        ],

                    "busy":
                        vehicle["Busy"],

                    "remaining_duration":
                        vehicle[
                            "Remaining Duration"
                        ],

                    "idle_time":
                        vehicle["Idle Time"]
                })

        # -------------------------------------------------
        # INCREASE TASK URGENCY
        # -------------------------------------------------

        for task in tasks_waiting_greedy:
            task["Urgency"] += (
                URGENCY_INCREMENT
            )

        for task in tasks_waiting_auction:
            task["Urgency"] += (
                URGENCY_INCREMENT
            )

        for task in tasks_waiting_qlearning:
            task["Urgency"] += (
                URGENCY_INCREMENT
            )

        # -------------------------------------------------
        # LOG WAITING TASKS
        # -------------------------------------------------

        for method_name, task_list in [

            ("Greedy", tasks_waiting_greedy),

            ("Auction", tasks_waiting_auction),

            ("Q-Learning", tasks_waiting_qlearning)
        ]:

            for task in task_list:

                waiting_task_log.append({

                    "time_step": t,

                    "method": method_name,

                    "task_id":
                        task["Task ID"],

                    "position":
                        task[
                            "Task Position (x, y)"
                        ],

                    "urgency":
                        task["Urgency"],

                    "duration":
                        task["Duration (min)"]
                })

        # -------------------------------------------------
        # DYNAMIC TASK ARRIVAL
        # -------------------------------------------------

        if random.random() < NEW_TASK_PROB:

            new_task = {

                "Task ID":
                    f"D{task_counter}",

                "Task Position (x, y)": (

                    random.randint(0, 100),

                    random.randint(0, 100)
                ),

                "Urgency":
                    random.randint(0, 9),

                "Duration (min)":
                    random.randint(10, 30)
            }

            dynamic_tasks_log.append({

                "time_step": t,

                **new_task
            })

            task_counter += 1

            tasks_waiting_greedy.append(
                copy.deepcopy(new_task)
            )

            tasks_waiting_auction.append(
                copy.deepcopy(new_task)
            )

            tasks_waiting_qlearning.append(
                copy.deepcopy(new_task)
            )

        # -------------------------------------------------
        # GREEDY
        # -------------------------------------------------

        if tasks_waiting_greedy:

            tasks_df = pd.DataFrame(
                tasks_waiting_greedy
            )

            alloc, details = (
                ga.greedy_allocation(
                    vehicles_greedy,
                    tasks_df
                )
            )

            allocations_greedy.update(
                alloc
            )

            engagement_metrics_greedy.extend(
                details
            )

            allocated_ids = set(
                alloc.keys()
            )

            tasks_waiting_greedy = [

                task

                for task in tasks_waiting_greedy

                if task["Task ID"]
                not in allocated_ids
            ]

        # -------------------------------------------------
        # AUCTION
        # -------------------------------------------------

        if tasks_waiting_auction:

            tasks_df = pd.DataFrame(
                tasks_waiting_auction
            )

            alloc, details = (
                aa.auction_allocation(
                    vehicles_auction,
                    tasks_df
                )
            )

            allocations_auction.update(
                alloc
            )

            engagement_metrics_auction.extend(
                details
            )

            allocated_ids = set(
                alloc.keys()
            )

            tasks_waiting_auction = [

                task

                for task in tasks_waiting_auction

                if task["Task ID"]
                not in allocated_ids
            ]

        # -------------------------------------------------
        # Q-LEARNING
        # -------------------------------------------------

        if tasks_waiting_qlearning:

            tasks_df = pd.DataFrame(
                tasks_waiting_qlearning
            )

            alloc, details = (
                qa.qlearning_allocation(
                    vehicles_qlearning,
                    tasks_df
                )
            )

            allocations_qlearning.update(
                alloc
            )

            engagement_metrics_qlearning.extend(
                details
            )

            allocated_ids = set(
                alloc.keys()
            )

            tasks_waiting_qlearning = [

                task

                for task in tasks_waiting_qlearning

                if task["Task ID"]
                not in allocated_ids
            ]

    # =====================================================
    # COMPUTE METRICS
    # =====================================================

    metrics_greedy = compute_metrics(

        engagement_metrics_greedy,

        vehicles_greedy,

        len(tasks_waiting_greedy)
    )

    metrics_auction = compute_metrics(

        engagement_metrics_auction,

        vehicles_auction,

        len(tasks_waiting_auction)
    )

    metrics_qlearning = compute_metrics(

        engagement_metrics_qlearning,

        vehicles_qlearning,

        len(tasks_waiting_qlearning)
    )

    all_results["Greedy"].append(
        metrics_greedy
    )

    all_results["Auction"].append(
        metrics_auction
    )

    all_results["Q-Learning"].append(
        metrics_qlearning
    )

    # =====================================================
    # SAVE RUN FILES
    # =====================================================

    save_dataframe(

        pd.DataFrame(dynamic_tasks_log),

        os.path.join(
            run_dir,
            "dynamic_tasks.csv"
        )
    )

    save_dataframe(

        pd.DataFrame(vehicle_state_log),

        os.path.join(
            run_dir,
            "vehicle_states.csv"
        )
    )

    save_dataframe(

        pd.DataFrame(waiting_task_log),

        os.path.join(
            run_dir,
            "waiting_tasks.csv"
        )
    )

    save_json(

        allocations_greedy,

        os.path.join(
            run_dir,
            "greedy_allocations.json"
        )
    )

    save_json(

        allocations_auction,

        os.path.join(
            run_dir,
            "auction_allocations.json"
        )
    )

    save_json(

        allocations_qlearning,

        os.path.join(
            run_dir,
            "qlearning_allocations.json"
        )
    )

    save_dataframe(

        pd.DataFrame(
            engagement_metrics_greedy
        ),

        os.path.join(
            run_dir,
            "greedy_engagement_metrics.csv"
        )
    )

    save_dataframe(

        pd.DataFrame(
            engagement_metrics_auction
        ),

        os.path.join(
            run_dir,
            "auction_engagement_metrics.csv"
        )
    )

    save_dataframe(

        pd.DataFrame(
            engagement_metrics_qlearning
        ),

        os.path.join(
            run_dir,
            "qlearning_engagement_metrics.csv"
        )
    )


# =========================================================
# AGGREGATE RESULTS
# =========================================================

metric_keys = [

    "total_tasks",

    "tasks_completed",

    "tasks_not_serviced",

    "avg_engagement_time",

    "avg_normalized_engagement",

    "energy_per_unit",

    "throughput",

    "total_idle_time",

    "avg_idle_time"
]

aggregate_rows = []

for method in all_results:

    for run_idx, metrics in enumerate(

        all_results[method]
    ):

        row = {

            "run_id":
                run_idx,

            "seed":
                GLOBAL_BASE_SEED
                + run_idx,

            "method":
                method
        }

        row.update(metrics)

        aggregate_rows.append(
            row
        )

aggregate_df = pd.DataFrame(
    aggregate_rows
)

save_dataframe(

    aggregate_df,

    os.path.join(
        RESULTS_DIR,
        "aggregate_metrics.csv"
    )
)

# =========================================================
# AVERAGE RESULTS
# =========================================================

average_results = {}

for method in all_results:

    average_results[method] = {}

    for key in metric_keys:

        values = [

            run[key]

            for run in all_results[
                method
            ]
        ]

        average_results[method][key] = (
            np.mean(values)
        )

summary_df = pd.DataFrame(
    average_results
).T

save_dataframe(

    summary_df,

    os.path.join(
        RESULTS_DIR,
        "aggregate_summary.csv"
    )
)

print("\n=================================================")
print("AVERAGED RESULTS ACROSS RUNS")
print("=================================================")

print(summary_df)

# =========================================================
# COMPARISON BAR PLOTS
# =========================================================

methods = [

    "Greedy",

    "Auction",

    "Q-Learning"
]

metrics_labels = [

    "tasks_completed",

    "tasks_not_serviced",

    "avg_engagement_time",

    "avg_normalized_engagement",

    "throughput",

    "energy_per_unit",

    "total_idle_time",

    "avg_idle_time"
]

titles = [

    "Tasks Completed\n(Higher Better)",

    "Tasks Not Serviced\n(Lower Better)",

    "Avg Engagement Time\n(Lower Better)",

    "Avg Normalized Engagement\n(Lower Better)",

    "Throughput\n(Higher Better)",

    "Energy per Unit Task\n(Lower Better)",

    "Total Idle Time\n(Lower Better)",

    "Avg Idle Time\n(Lower Better)"
]

fig, axes = plt.subplots(

    2,

    4,

    figsize=(24, 10)
)

axes = axes.flatten()

for i, metric in enumerate(
    metrics_labels
):

    values = [

        average_results[
            "Greedy"
        ][metric],

        average_results[
            "Auction"
        ][metric],

        average_results[
            "Q-Learning"
        ][metric]
    ]

    axes[i].bar(
        methods,
        values
    )

    axes[i].set_title(
        titles[i]
    )

    axes[i].set_ylabel(
        metric
    )

plt.tight_layout()

comparison_plot_path = os.path.join(

    RESULTS_DIR,

    "comparison_metrics.png"
)

plt.savefig(

    comparison_plot_path,

    dpi=300,

    bbox_inches="tight"
)

plt.close()

# =========================================================
# DISTRIBUTION PLOTS (2x4)
# =========================================================

fig, axes = plt.subplots(2, 4, figsize=(24, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics_labels):

    axes[i].hist([r[metric] for r in all_results["Greedy"]], alpha=0.5, label="Greedy")
    axes[i].hist([r[metric] for r in all_results["Auction"]], alpha=0.5, label="Auction")
    axes[i].hist([r[metric] for r in all_results["Q-Learning"]], alpha=0.5, label="Q-Learning")

    axes[i].set_title(metric)
    axes[i].legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "comparison_distributions.png"), dpi=300)
plt.close()

# =========================================================
# TREND PLOTS (2x4)
# =========================================================

fig, axes = plt.subplots(2, 4, figsize=(24, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics_labels):

    axes[i].plot([r[metric] for r in all_results["Greedy"]], label="Greedy")
    axes[i].plot([r[metric] for r in all_results["Auction"]], label="Auction")
    axes[i].plot([r[metric] for r in all_results["Q-Learning"]], label="Q-Learning")

    axes[i].set_title(metric)
    axes[i].legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "comparison_trends.png"), dpi=300)
plt.close()

print("\n=================================================")
print("ALL RESULTS SAVED")
print("=================================================")

print(f"Results directory:\n{RESULTS_DIR}")