import numpy as np
import random
from collections import deque

class TabularFleetLearner:
    """
    Architectural Redesign: Decentralized Independent Q-Learning (IQL) with Shared Experience.
    Vehicles independently evaluate states and select actions locally, completely 
    decoupled from a central auction framework.
    """
    def __init__(self, alpha=0.1, gamma=0.9, temp=1.0):
        self.q_table = {} # (state, action) -> q_value
        self.alpha = alpha
        self.gamma = gamma
        self.temp = temp # Boltzmann Exploration Temperature
        self.replay_buffer = deque(maxlen=1000)
        self.batch_size = 32
        
    def get_local_state(self, vehicle, broadcast_tasks, radar_idle_pos, current_time):
        """
        DECENTRALIZED STATE REPRESENTATION:
        Restricts vehicle sight to a realistic local sensor/communication radius (150m).
        """
        v_pos = vehicle['Vehicle Position (x, y)']
        
        # 1. Internal Grid Mapping
        gx = int(min(2, max(0, v_pos[0] // 167)))
        gy = int(min(2, max(0, v_pos[1] // 167)))
        grid = gx * 3 + gy
        
        sensor_radius = 150.0
        
        # 2. Localized Demand (Only tasks within immediate sensor range)
        local_demand = sum(1 for t in broadcast_tasks if np.sqrt((v_pos[0]-t[0])**2 + (v_pos[1]-t[1])**2) < sensor_radius)
        demand_bin = 1 if local_demand > 1 else 0
        
        # 3. Localized Competition (Other idle vehicles within sensor range)
        local_supply = sum(1 for p in radar_idle_pos if np.sqrt((v_pos[0]-p[0])**2 + (v_pos[1]-p[1])**2) < sensor_radius) - 1
        supply_bin = 1 if local_supply > 1 else 0
        
        # 4. Temporal factor
        phase = 1 if current_time > 300 else 0
        
        return (grid, demand_bin, supply_bin, phase)

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def select_action(self, state):
        qs = [self.get_q(state, a) for a in range(4)]
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

# Shared policy repository for independent, homogeneous agents
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
    
    # Shuffle processing sequence to simulate randomized local transmission order
    random.shuffle(idle_indices)
    
    for v_idx in idle_indices:
        vehicle = vehicles_df.loc[v_idx]
        
        # 1. Independent Local State Evaluation
        state = _fleet_agent.get_local_state(vehicle, task_pos, idle_pos, current_time)
        
        # 2. Independent Q-table Action Selection
        action = _fleet_agent.select_action(state)
        
        # 3. Filter out tasks already claimed by earlier peers in the broadcast cycle
        available_tasks = [t for t in tasks if t['Task ID'] not in assigned_tasks]
        if not available_tasks: 
            break
            
        # Evaluate local candidates using the chosen behavior profile
        scored = []
        for t in available_tasks:
            dist = np.sqrt((vehicle['Vehicle Position (x, y)'][0]-t['Task Position (x, y)'][0])**2 + (vehicle['Vehicle Position (x, y)'][1]-t['Task Position (x, y)'][1])**2)
            tt = dist / vehicle['Speed']
            tot = t['Duration (min)'] + tt
            
            if vehicle['Battery Level (%)'] < tot: 
                continue  # Local battery safety cutoff
            
            # Action Space mapping: Archetype heuristics
            if action == 1:    # EFFICIENCY
                score = t['Duration (min)'] / (tt + 0.1)
            elif action == 2:  # URGENCY
                score = t['Urgency'] * 2.0 - tt
            elif action == 3:  # POSITIONING
                rad = 180.0
                loc_dens = sum(1 for t2 in task_pos if np.sqrt((t['Task Position (x, y)'][0]-t2[0])**2 + (t['Task Position (x, y)'][1]-t2[1])**2) < rad)
                score = loc_dens * 5.0 - tt
            else:              # Action 0: WAIT/STAY
                score = -1.0   # Baseline for ignoring tasks
            
            scored.append((score, t, tt, tot))

        # 4. Independent Execution & Update
        if action > 0 and scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            _, best_t, b_tt, b_tot = scored[0]
            
            reward = 1.0 + (best_t['Duration (min)'] / b_tot) + (best_t['Urgency'] / 10.0) - (b_tt / 20.0)
            
            # Local Next-State Prediction
            next_v = vehicle.copy()
            next_v['Vehicle Position (x, y)'] = best_t['Task Position (x, y)']
            next_state = _fleet_agent.get_local_state(next_v, task_pos, idle_pos, current_time)
            
            # Store experience locally to transition to training
            _fleet_agent.push(state, action, reward, next_state)
            
            # Environment assignment changes
            allocations[best_t['Task ID']] = vehicle['Vehicle ID']
            vehicles_df.at[v_idx, 'Battery Level (%)'] -= b_tot
            vehicles_df.at[v_idx, 'Busy'] = True
            vehicles_df.at[v_idx, 'Remaining Duration'] = float(b_tot)
            vehicles_df.at[v_idx, 'Vehicle Position (x, y)'] = best_t['Task Position (x, y)']
            
            assigned_tasks.add(best_t['Task ID'])
            engagement_details.append({
                "task_id": best_t['Task ID'], "task_duration": best_t['Duration (min)'],
                "travel_time": b_tt, "engagement_time": b_tot,
                "normalized_engagement_time": b_tot / best_t['Duration (min)'], "energy_consumed": b_tot
            })
        else:
            # Action 0 or fallback state (No valid task found)
            reward = -1.0 if state[1] == 1 else -0.05
            _fleet_agent.push(state, 0, reward, state)

    # Centralized training phase updates the shared Q-table parameters
    _fleet_agent.train_batch()
    
    return allocations, engagement_details
