import random
import numpy as np
import pandas as pd

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def generate_vehicle_data(num_vehicles):
    vehicles = []
    for i in range(num_vehicles):
        vehicle_id = f"V{i+1}"
        position = (random.randint(0, 100), random.randint(0, 100))
        battery_level = random.randint(50, 100)
        speed = random.randint(3, 10)

        vehicles.append({
            'Vehicle ID': vehicle_id,
            'Vehicle Position (x, y)': position,
            'Battery Level (%)': float(battery_level),
            'Speed': float(speed)
        })

    return pd.DataFrame(vehicles)

def generate_task_data(num_tasks):
    tasks = []
    for i in range(num_tasks):
        task_id = f"T{i+1}"
        position = (random.randint(0, 100), random.randint(0, 100))
        urgency = random.randint(0, 9)
        duration = random.randint(10, 30)

        tasks.append({
            'Task ID': task_id,
            'Task Position (x, y)': position,
            'Urgency': urgency,
            'Duration (min)': float(duration)
        })

    return pd.DataFrame(tasks)