import numpy as np
import random
from collections import deque

class TabularFleetLearner:
    """
    Architectural Redesign: Tabular Q-Learning with Shared Experience.
    Solves the 'Sparse Visitation' and 'Non-Stationarity' by using a shared 
    policy for all homogeneous vehicles and a replay buffer to decorrelate updates.
    """
    def __init__(self, alpha=0.1, gamma=0.9, temp=1.0):
        self.q_table = {} # (state, action) -> q_value
        self.alpha = alpha
        self.gamma = gamma
        self.temp = temp # Boltzmann Heat
        self.replay_buffer = deque(maxlen=1000)
        self.batch_size = 32
        
    def get_state(self, vehicle, tasks, idle_pos, current_time):
        """
        Discretized State Space (144 states):
        - Grid (9)
        - Demand High/Low (2)
        - Supply High/Low (2)
        - Idle Time High/Low (2)
        - Phase Early/Late (2)
        """
        v_pos = vehicle['Vehicle Position (x, y)']
        gx = int(min(2, max(0, v_pos[0] // 167)))
        gy = int(min(2, max(0, v_pos[1] // 167)))
        grid = gx * 3 + gy
        
        radius = 160.0 # Slightly larger radius for awareness
        demand = sum(1 for t in tasks if np.sqrt((v_pos[0]-t[0])**2 + (v_pos[1]-t[1])**2) < radius)
        demand_bin = 1 if demand > 0 else 0
        
        supply = sum(1 for p in idle_pos if np.sqrt((v_pos[0]-p[0])**2 + (v_pos[1]-p[1])**2) < radius)
        supply_bin = 1 if supply > 1 else 0
        
        # Idle Time Bin: Crucial for reducing stagnation
        idle_time = vehicle.get('Idle Time', 0)
        idle_bin = 1 if idle_time > 15 else 0
        
        phase = 1 if current_time > 300 else 0
        
        return (grid, demand_bin, supply_bin, idle_bin, phase)

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.5) # Optimistic Initialization

    def select_action(self, state):
        qs = [self.get_q(state, a) for a in range(4)]
        # Boltzmann Selection with cooling
        exp_qs = np.exp((np.array(qs) - np.max(qs)) / self.temp)
        probs = exp_qs / exp_qs.sum()
        return np.random.choice(4, p=probs)

    def push(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))

    def train_batch(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        for s, a, r, s_next in batch:
            current_q = self.get_q(s, a)
            max_next_q = max([self.get_q(s_next, a_next) for a_next in range(4)])
            target = r + self.gamma * max_next_q
            self.q_table[(s, a)] = current_q + self.alpha * (target - current_q)

# Global Homogeneous Agent
_fleet_agent = TabularFleetLearner()

def qlearning_allocation(vehicles_df, tasks_df, current_time=0):
    global _fleet_agent
    allocations = {}
    engagement_details = []
    
    idle_indices = vehicles_df[~vehicles_df['Busy']].index.tolist()
    if not idle_indices or tasks_df.empty:
        return allocations, engagement_details

    tasks = tasks_df.to_dict('records')
    task_pos = [t['Task Position (x, y)'] for t in tasks]
    idle_pos = [vehicles_df.at[i, 'Vehicle Position (x, y)'] for i in idle_indices]
    
    assigned_tasks = set()
    random.shuffle(idle_indices)
    
    for v_idx in idle_indices:
        vehicle = vehicles_df.loc[v_idx]
        state = _fleet_agent.get_state(vehicle, task_pos, idle_pos, current_time)
        
        action = _fleet_agent.select_action(state)
        
        available_tasks = [t for t in tasks if t['Task ID'] not in assigned_tasks]
        if not available_tasks: break
            
        # Action-specific candidate scoring
        scored = []
        for t in available_tasks:
            dist = np.sqrt((vehicle['Vehicle Position (x, y)'][0]-t['Task Position (x, y)'][0])**2 + (vehicle['Vehicle Position (x, y)'][1]-t['Task Position (x, y)'][1])**2)
            tt = dist / vehicle['Speed']
            tot = t['Duration (min)'] + tt
            
            if vehicle['Battery Level (%)'] < tot: continue
            
            if action == 1: # EFFICIENCY: Maximize work density
                score = t['Duration (min)'] / (tt + 0.1)
            elif action == 2: # URGENCY: Prioritize critical
                score = t['Urgency'] * 2.5 - tt
            elif action == 3: # POSITIONING: Focus on future high-demand zones
                rad = 200.0
                loc_dens = sum(1 for t2 in task_pos if np.sqrt((t['Task Position (x, y)'][0]-t2[0])**2 + (t['Task Position (x, y)'][1]-t2[1])**2) < rad)
                # Bias towards regions with many tasks but few vehicles
                score = loc_dens * 4.0 - tt
            else: # Action 0: WAIT (Neutral)
                score = -100 # Discourage wait if tasks are available
            
            scored.append((score, t, tt, tot))

        # Selection & Execution
        if action > 0 and scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            _, best_t, b_tt, b_tot = scored[0]
            
            # Reward Design: 
            # Focus on Work done + Speed of response + Recovery from long idleness
            productivity = best_t['Duration (min)'] / b_tot
            urgency_score = best_t['Urgency'] / 8.0
            idleness_recovery = 1.0 if state[3] == 1 else 0.0 # Reward taking task after long idle
            
            reward = 1.5 * productivity + urgency_score + idleness_recovery - (b_tt / 15.0)
            
            # Post-state estimation
            next_v = vehicle.copy()
            next_v['Vehicle Position (x, y)'] = best_t['Task Position (x, y)']
            # Approximate next idle time as 0 after finishing task
            next_v['Idle Time'] = 0
            next_state = _fleet_agent.get_state(next_v, task_pos, idle_pos, current_time)
            
            _fleet_agent.push(state, action, reward, next_state)
            
            allocations[best_t['Task ID']] = vehicle['Vehicle ID']
            vehicles_df.at[v_idx, 'Battery Level (%)'] -= b_tot
            vehicles_df.at[v_idx, 'Busy'] = True
            vehicles_df.at[v_idx, 'Remaining Duration'] = float(b_tot)
            vehicles_df.at[v_idx, 'Vehicle Position (x, y)'] = best_t['Task Position (x, y)']
            vehicles_df.at[v_idx, 'Idle Time'] = 0 # Reset idle time
            
            assigned_tasks.add(best_t['Task ID'])
            engagement_details.append({
                "task_id": best_t['Task ID'], "task_duration": best_t['Duration (min)'],
                "travel_time": b_tt, "engagement_time": b_tot,
                "normalized_engagement_time": b_tot / best_t['Duration (min)'], "energy_consumed": b_tot
            })
        else:
            # Action 0 or no candidate: Idle penalty
            # Penalty increases with duration of idleness and local demand
            idle_duration = vehicle.get('Idle Time', 0)
            penalty = -0.5 - (0.05 * idle_duration)
            if state[1] == 1: # High local demand
                penalty -= 2.0
            
            _fleet_agent.push(state, 0, penalty, state)

    # Batch learning step
    _fleet_agent.train_batch()
    # Gradual cooling
    _fleet_agent.temp = max(0.05, _fleet_agent.temp * 0.999)
    
    return allocations, engagement_details
