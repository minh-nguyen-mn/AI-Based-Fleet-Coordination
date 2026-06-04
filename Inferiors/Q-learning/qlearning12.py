import numpy as np
import pandas as pd

class DecentralizedVehicleAgent:
    """
    Decentralized Tabular Q-Learner representing an individual autonomous vehicle.
    Uses a standard Epsilon-Greedy strategy driven by an episode-level decay schedule.
    """
    def __init__(self, vehicle_id, alpha=0.1, gamma=0.9):
        self.vehicle_id = vehicle_id
        self.q_table = {}  # Map: (state, action) -> q_value
        self.alpha = alpha
        self.gamma = gamma
        
    def get_local_state(self, vehicle, tasks_df, idle_positions):
        """
        Discretized State Space from the local vehicle's perspective (36 states):
        - Grid Location (0-8)
        - Local Demand Density (0: Low, 1: High)
        - Local Competition/Supply (0: Low, 1: High)
        """
        v_pos = vehicle['Vehicle Position (x, y)']
        
        # 1. Spatial Discretization (3x3 Grid over 100x100 space)
        grid_size = 100 / 3
        gx = int(min(2, max(0, v_pos[0] // grid_size)))
        gy = int(min(2, max(0, v_pos[1] // grid_size)))
        grid = gx * 3 + gy 
        
        # 2. Local Demand (Count tasks within bounded 25-unit radius)
        radius = 25.0
        demand_count = 0
        if not tasks_df.empty:
            for _, t in tasks_df.iterrows():
                t_pos = t['Task Position (x, y)']
                if np.sqrt((v_pos[0] - t_pos[0])**2 + (v_pos[1] - t_pos[1])**2) < radius:
                    demand_count += 1
        demand_bin = 1 if demand_count > 1 else 0
        
        # 3. Local Supply (Count competing idle vehicles within 25-unit radius)
        supply_count = 0
        for p in idle_positions:
            if np.sqrt((v_pos[0] - p[0])**2 + (v_pos[1] - p[1])**2) < radius:
                supply_count += 1
        # Subtract self-count to track true external local competition
        supply_bin = 1 if (supply_count - 1) > 2 else 0
        
        return (grid, demand_bin, supply_bin)

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def select_action(self, state, epsilon):
        """
        Standard Epsilon-Greedy Action Selection.
        """
        if np.random.rand() < epsilon:
            return np.random.choice(4)
        
        qs = [self.get_q(state, a) for a in range(4)]
        max_q = np.max(qs)
        # Random tie-breaking for equal Q-values
        actions_with_max_q = [a for a, q in enumerate(qs) if q == max_q]
        return np.random.choice(actions_with_max_q)

    def update_q(self, state, action, reward, next_state):
        """Standard online Tabular Q-learning update rule."""
        current_q = self.get_q(state, action)
        max_next_q = max([self.get_q(next_state, a_next) for a_next in range(4)])
        target = reward + self.gamma * max_next_q
        self.q_table[(state, action)] = current_q + self.alpha * (target - current_q)


# Persistent global registry preserving decentralized agent states across episodes
_decentralized_fleet_registry = {}

def qlearning_allocation(vehicles_df, tasks_df, epsilon=0.0):
    """
    Decentralized Execution Flow driven by standard epsilon-greedy selection.
    """
    global _decentralized_fleet_registry
    allocations = {}
    engagement_details = []
    
    idle_indices = vehicles_df[vehicles_df['Busy'] == False].index.tolist()
    if not idle_indices or tasks_df.empty:
        return allocations, engagement_details

    # Initialize agents mapping to physical IDs
    for v_idx in idle_indices:
        v_id = vehicles_df.at[v_idx, 'Vehicle ID']
        if v_id not in _decentralized_fleet_registry:
            _decentralized_fleet_registry[v_id] = DecentralizedVehicleAgent(vehicle_id=v_id)

    idle_positions = [vehicles_df.at[i, 'Vehicle Position (x, y)'] for i in idle_indices]
    vehicle_intents = {}

    # Step 1 & 2: Local Perception & Intent Selection
    for v_idx in idle_indices:
        vehicle = vehicles_df.loc[v_idx]
        v_id = vehicle['Vehicle ID']
        agent = _decentralized_fleet_registry[v_id]
        
        state = agent.get_local_state(vehicle, tasks_df, idle_positions)
        action = agent.select_action(state, epsilon)
        
        vehicle_intents[v_idx] = {
            'agent': agent,
            'state': state,
            'action': action,
            'targeted_task': None,
            'metrics': None
        }
        
        if action == 0:
            reward = -1.0 if state[1] == 1 else -0.05
            agent.update_q(state, 0, reward, state)
            continue

        candidate_scores = []
        for _, task in tasks_df.iterrows():
            dist = np.sqrt((vehicle['Vehicle Position (x, y)'][0] - task['Task Position (x, y)'][0])**2 + 
                           (vehicle['Vehicle Position (x, y)'][1] - task['Task Position (x, y)'][1])**2)
            travel_time = dist / vehicle['Speed']
            total_time = task['Duration (min)'] + travel_time
            
            if vehicle['Battery Level (%)'] < total_time:
                continue 
                
            if action == 1:    # CLOSEST
                score = -travel_time
            elif action == 2:  # URGENCY
                score = task['Urgency'] * 10.0 - travel_time
            elif action == 3:  # EFFICIENCY
                score = task['Duration (min)'] / (total_time + 0.1)
            else:
                score = 0
                
            candidate_scores.append((score, task, travel_time, total_time))
            
        if candidate_scores:
            candidate_scores.sort(key=lambda x: x[0], reverse=True)
            _, best_task, b_tt, b_tot = candidate_scores[0]
            vehicle_intents[v_idx]['targeted_task'] = best_task['Task ID']
            vehicle_intents[v_idx]['metrics'] = (best_task, b_tt, b_tot)

    # Step 3 & 4: Conflict Resolution
    task_claims = {}
    for v_idx, intent in vehicle_intents.items():
        t_id = intent['targeted_task']
        if t_id is not None:
            if t_id not in task_claims:
                task_claims[t_id] = []
            task_claims[t_id].append(v_idx)

    for t_id, claiming_v_indices in task_claims.items():
        if len(claiming_v_indices) == 1:
            winner_idx = claiming_v_indices[0]
        else:
            winner_idx = min(claiming_v_indices, key=lambda idx: np.sqrt(
                (vehicles_df.at[idx, 'Vehicle Position (x, y)'][0] - tasks_df.loc[tasks_df['Task ID'] == t_id, 'Task Position (x, y)'].values[0][0])**2 +
                (vehicles_df.at[idx, 'Vehicle Position (x, y)'][1] - tasks_df.loc[tasks_df['Task ID'] == t_id, 'Task Position (x, y)'].values[0][1])**2
            ))
            
        intent = vehicle_intents[winner_idx]
        agent = intent['agent']
        state = intent['state']
        action = intent['action']
        best_task, b_tt, b_tot = intent['metrics']
        
        reward = 2.0 + (best_task['Urgency'] * 0.5) - (b_tt * 0.1)
        
        next_v = vehicles_df.loc[winner_idx].copy()
        next_v['Vehicle Position (x, y)'] = best_task['Task Position (x, y)']
        next_state = agent.get_local_state(next_v, tasks_df, idle_positions)
        
        agent.update_q(state, action, reward, next_state)
        
        allocations[t_id] = vehicles_df.at[winner_idx, 'Vehicle ID']
        vehicles_df.at[winner_idx, 'Battery Level (%)'] -= b_tot
        vehicles_df.at[winner_idx, 'Busy'] = True
        vehicles_df.at[winner_idx, 'Remaining Duration'] = float(b_tot)
        vehicles_df.at[winner_idx, 'Vehicle Position (x, y)'] = best_task['Task Position (x, y)']
        
        engagement_details.append({
            "task_id": t_id, "task_duration": best_task['Duration (min)'],
            "travel_time": b_tt, "engagement_time": b_tot,
            "normalized_engagement_time": b_tot / best_task['Duration (min)'], "energy_consumed": b_tot
        })
        
        for loser_idx in claiming_v_indices:
            if loser_idx != winner_idx:
                l_intent = vehicle_intents[loser_idx]
                l_intent['agent'].update_q(l_intent['state'], l_intent['action'], -0.5, l_intent['state'])

    return allocations, engagement_details