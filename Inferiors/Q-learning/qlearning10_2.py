import numpy as np
import pandas as pd
import random

class DecentralizedVehicleAgent:
    """
    Decentralized Tabular Q-Learner representing an individual autonomous vehicle.
    Each vehicle maintains its own local perspective, action logic, and Q-table.
    """
    def __init__(self, vehicle_id, alpha=0.1, gamma=0.9, temp=1.0):
        self.vehicle_id = vehicle_id
        self.q_table = {}  # Map: (state, action) -> q_value
        self.alpha = alpha
        self.gamma = gamma
        self.temp = temp   # Temperature for Boltzmann exploration
        
    def get_local_state(self, vehicle, tasks_df, idle_positions, current_time):
        """
        Discretized State Space from the local vehicle's perspective (72 states):
        - Grid Location (0-8)
        - Local Demand Density (0: Low, 1: High)
        - Local Competition/Supply (0: Low, 1: High)
        - Temporal Phase (0: Early, 1: Late)
        """
        v_pos = vehicle['Vehicle Position (x, y)']
        
        # 1. Spatial Discretization (3x3 Grid)
        grid_size = 100 / 3
        gx = int(min(2, max(0, v_pos[0] // grid_size)))
        gy = int(min(2, max(0, v_pos[1] // grid_size)))

        # Clamp exactly to 2 to prevent index out of bounds
        gx = min(gx, 2)
        gy = min(gy, 2)

        grid = gx * 3 + gy 
        
        # 2. Local Demand (Count tasks within radius)
        radius = 25.0
        demand_count = 0
        if not tasks_df.empty:
            for _, t in tasks_df.iterrows():
                t_pos = t['Task Position (x, y)']
                if np.sqrt((v_pos[0] - t_pos[0])**2 + (v_pos[1] - t_pos[1])**2) < radius:
                    demand_count += 1
        demand_bin = 1 if demand_count > 1 else 0
        
        # 3. Local Supply (Count competing idle vehicles within radius)
        supply_count = 0
        for p in idle_positions:
            if np.sqrt((v_pos[0] - p[0])**2 + (v_pos[1] - p[1])**2) < radius:
                supply_count += 1
        # Subtract 1 so the vehicle doesn't count itself as competition
        supply_bin = 1 if (supply_count - 1) > 2 else 0
        
        # 4. Temporal Phase
        phase = 1 if current_time > 50 else 0
        
        return (grid, demand_bin, supply_bin, phase)

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def select_action(self, state):
        """
        Exploration Strategy: Boltzmann (Softmax) distribution over the 4 choices.
        Action Spaces:
        0: STAY IDLE / Refuse allocation
        1: SELECT CLOSEST Task
        2: SELECT HIGHEST URGENCY Task
        3: SELECT MOST ENERGY-EFFICIENT Task
        """
        qs = [self.get_q(state, a) for a in range(4)]
        # Subtract max for numerical stability in softmax
        exp_qs = np.exp((np.array(qs) - np.max(qs)) / self.temp)
        probs = exp_qs / exp_qs.sum()
        return np.random.choice(4, p=probs)

    def update_q(self, state, action, reward, next_state):
        """Standard online Tabular Q-learning update rule."""
        current_q = self.get_q(state, action)
        max_next_q = max([self.get_q(next_state, a_next) for a_next in range(4)])
        target = reward + self.gamma * max_next_q
        self.q_table[(state, action)] = current_q + self.alpha * (target - current_q)


# Persistent global registry preserving decentralized agent states across evaluation steps
_decentralized_fleet_registry = {}

def qlearning_allocation(vehicles_df, tasks_df, current_time=0):
    """
    Decentralized Execution Flow:
    1. Each idle vehicle assesses its local state independently.
    2. Each vehicle selects an operational intent/action via its own Q-table.
    3. Vehicles independently select their target tasks based on their intent.
    4. Conflict resolution occurs if multiple vehicles claim the same asset.
    """
    global _decentralized_fleet_registry
    allocations = {}
    engagement_details = []
    
    idle_indices = vehicles_df[~vehicles_df['Busy']].index.tolist()
    if not idle_indices or tasks_df.empty:
        return allocations, engagement_details

    # Initialize agents dynamically if they don't exist yet
    for v_idx in idle_indices:
        v_id = vehicles_df.at[v_idx, 'Vehicle ID']
        if v_id not in _decentralized_fleet_registry:
            _decentralized_fleet_registry[v_id] = DecentralizedVehicleAgent(vehicle_id=v_id)

    # Collect all current idle positions for local coordination metrics
    idle_positions = [vehicles_df.at[i, 'Vehicle Position (x, y)'] for i in idle_indices]
    
    # Store decision contexts to perform clean post-conflict Q-updates
    vehicle_intents = {}

    # Step 1 & 2: Local Perception & Intent Selection (Decentralized Loop)
    for v_idx in idle_indices:
        vehicle = vehicles_df.loc[v_idx]
        v_id = vehicle['Vehicle ID']
        agent = _decentralized_fleet_registry[v_id]
        
        # Observe local state and select intent action
        state = agent.get_local_state(vehicle, tasks_df, idle_positions, current_time)
        action = agent.select_action(state)
        
        vehicle_intents[v_idx] = {
            'agent': agent,
            'state': state,
            'action': action,
            'targeted_task': None,
            'metrics': None
        }
        
        if action == 0:
            # Intent is to stay idle. Immediate local penalty if tasks are neglected nearby.
            reward = -1.0 if state[1] == 1 else -0.05
            agent.update_q(state, 0, reward, state)
            continue

        # Evaluate and score tasks based on the specific chosen action strategy
        candidate_scores = []
        for _, task in tasks_df.iterrows():
            dist = np.sqrt((vehicle['Vehicle Position (x, y)'][0] - task['Task Position (x, y)'][0])**2 + 
                           (vehicle['Vehicle Position (x, y)'][1] - task['Task Position (x, y)'][1])**2)
            travel_time = dist / vehicle['Speed']
            total_time = task['Duration (min)'] + travel_time
            
            if vehicle['Battery Level (%)'] < total_time:
                continue  # Infeasible due to energy constraints
                
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

    # Step 3 & 4: Conflict Resolution (Simulating decentralized contention)
    # Group vehicle claims by Task ID
    task_claims = {}
    for v_idx, intent in vehicle_intents.items():
        t_id = intent['targeted_task']
        if t_id is not None:
            if t_id not in task_claims:
                task_claims[t_id] = []
            task_claims[t_id].append(v_idx)

    # Resolve contentions (e.g., if two vehicles choose the same task, the closest wins)
    for t_id, claiming_v_indices in task_claims.items():
        if len(claiming_v_indices) == 1:
            winner_idx = claiming_v_indices[0]
        else:
            # Conflict resolution rule: closest vehicle to the asset gets it
            winner_idx = min(claiming_v_indices, key=lambda idx: np.sqrt(
                (vehicles_df.at[idx, 'Vehicle Position (x, y)'][0] - tasks_df.loc[tasks_df['Task ID'] == t_id, 'Task Position (x, y)'].values[0][0])**2 +
                (vehicles_df.at[idx, 'Vehicle Position (x, y)'][1] - tasks_df.loc[tasks_df['Task ID'] == t_id, 'Task Position (x, y)'].values[0][1])**2
            ))
            
        # Execute assignment for the winner
        intent = vehicle_intents[winner_idx]
        agent = intent['agent']
        state = intent['state']
        action = intent['action']
        best_task, b_tt, b_tot = intent['metrics']
        
        # Calculate Reward based on task performance metrics
        reward = 2.0 + (best_task['Urgency'] * 0.5) - (b_tt * 0.1)
        
        # Update winning vehicle's telemetry data
        next_v = vehicles_df.loc[winner_idx].copy()
        next_v['Vehicle Position (x, y)'] = best_task['Task Position (x, y)']
        next_state = agent.get_local_state(next_v, tasks_df, idle_positions, current_time)
        
        # Learn from success
        agent.update_q(state, action, reward, next_state)
        
        # Apply state changes to baseline data frame
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
        
        # Penalize the losers who lost the conflict (simulates real decentralized friction)
        for loser_idx in claiming_v_indices:
            if loser_idx != winner_idx:
                l_intent = vehicle_intents[loser_idx]
                # Lost conflict penalty: wasted negotiation/idle time step
                l_intent['agent'].update_q(l_intent['state'], l_intent['action'], -0.5, l_intent['state'])

    return allocations, engagement_details