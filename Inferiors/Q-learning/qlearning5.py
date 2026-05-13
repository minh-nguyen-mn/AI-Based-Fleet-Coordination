import numpy as np
import random

class TabularQLearner:
    def __init__(self, alpha=0.5, gamma=0.8, epsilon=0.1):
        self.q_table = {}  # (state, action_id)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.98
        self.min_epsilon = 0.02
        
    def get_state(self, v_row, tasks, idle_vehicles):
        """
        State: (Location, Demand_Bin, Supply_Bin, Under_Utilized)
        Under_Utilized helps the agent realize when it's falling behind.
        """
        pos = v_row['Vehicle Position (x, y)']
        gx, gy = int(pos[0]/167), int(pos[1]/167)
        gx, gy = min(2, gx), min(2, gy)
        loc = gx * 3 + gy
        
        radius = 180.0
        # Demand & Supply
        demand = sum(1 for t in tasks if np.sqrt((pos[0]-t['Task Position (x, y)'][0])**2 + (pos[1]-t['Task Position (x, y)'][1])**2) < radius)
        demand_bin = 1 if demand > 1 else 0
        
        supply = sum(1 for p in idle_vehicles if np.sqrt((pos[0]-p[0])**2 + (pos[1]-p[1])**2) < radius) - 1
        supply_bin = 1 if supply > 1 else 0
        
        # Under-utilization bin (based on idle time)
        under_util = 1 if v_row.get('Idle Time', 0) > 30 else 0
        
        return (loc, demand_bin, supply_bin, under_util)

    def get_q(self, state, action):
        if (state, action) not in self.q_table:
            # Neutral initialization
            self.q_table[(state, action)] = 0.5 
        return self.q_table[(state, action)]

    def update(self, state, action, reward, next_state):
        current_q = self.get_q(state, action)
        next_max_q = max([self.get_q(next_state, a) for a in range(4)])
        target = reward + self.gamma * next_max_q
        self.q_table[(state, action)] = current_q + self.alpha * (target - current_q)

# Global Agent
_agent = TabularQLearner()

def qlearning_allocation(vehicles_df, tasks_df, current_time=0):
    global _agent
    allocations = {}
    engagement_details = []
    
    idle_indices = vehicles_df[~vehicles_df['Busy']].index.tolist()
    if not idle_indices or tasks_df.empty:
        return allocations, engagement_details

    tasks = tasks_df.to_dict('records')
    idle_pos = [vehicles_df.at[i, 'Vehicle Position (x, y)'] for i in idle_indices]
    
    random.shuffle(idle_indices)
    assigned_tasks = set()
    
    for v_idx in idle_indices:
        vehicle = vehicles_df.loc[v_idx]
        v_state = _agent.get_state(vehicle, tasks, idle_pos)
        
        # Action selection
        if random.random() < _agent.epsilon:
            action = random.randint(0, 3)
        else:
            qs = [_agent.get_q(v_state, a) for a in range(4)]
            action = np.argmax(qs)
            
        available_tasks = [t for t in tasks if t['Task ID'] not in assigned_tasks]
        
        if action > 0 and available_tasks:
            scored = []
            for t in available_tasks:
                v_pos = vehicle['Vehicle Position (x, y)']
                t_pos = t['Task Position (x, y)']
                dist = np.sqrt((v_pos[0]-t_pos[0])**2 + (v_pos[1]-t_pos[1])**2)
                tt = dist / vehicle['Speed']
                tot = t['Duration (min)'] + tt
                
                # REJECTION FILTER: Dynamic based on battery and productivity
                # If battery is low, stay close (< 30% of duration). 
                # If battery is high, can travel more (< 80% of duration).
                batt_p = vehicle['Battery Level (%)'] / 100.0
                max_overhead = 0.3 + (batt_p * 0.6)
                
                if tt > (t['Duration (min)'] * max_overhead) and t['Urgency'] < 18:
                    continue
                
                if vehicle['Battery Level (%)'] < tot: continue
                
                # Archetype Scorer
                if action == 1: # URGENCY
                    score = (t['Urgency'] * 4.0) + (t['Duration (min)'] / (tt + 2.0))
                elif action == 2: # EFFICIENCY
                    score = (t['Duration (min)'] / (tt + 0.1))
                else: # STRATEGIC
                    radius = 200.0
                    dens = sum(1 for t2 in tasks if np.sqrt((t_pos[0]-t2['Task Position (x, y)'][0])**2 + (t_pos[1]-t2['Task Position (x, y)'][1])**2) < radius)
                    score = (dens * 3.0) + (t['Duration (min)'] / (tt + 1.0))
                
                scored.append((score, t, tt, tot))
                
            if scored:
                scored.sort(key=lambda x: x[0], reverse=True)
                _, b_task, b_tt, b_tot = scored[0]
                
                # REWARD: Thru + Energy + Reliability
                productivity = b_task['Duration (min)'] / b_tot
                urg_bonus = (b_task['Urgency'] / 15.0) 
                
                # Utilization Recovery reward
                recovery = 1.0 if v_state[3] == 1 else 0.0
                
                reward = (productivity * 4.0) + urg_bonus + recovery - (b_tt / 25.0)
                
                next_v = vehicle.copy()
                next_v['Vehicle Position (x, y)'] = b_task['Task Position (x, y)']
                next_state = _agent.get_state(next_v, tasks, idle_pos)
                _agent.update(v_state, action, reward, next_state)
                
                # Execution
                allocations[b_task['Task ID']] = vehicle['Vehicle ID']
                vehicles_df.at[v_idx, 'Battery Level (%)'] -= b_tot
                vehicles_df.at[v_idx, 'Busy'] = True
                vehicles_df.at[v_idx, 'Remaining Duration'] = float(b_tot)
                vehicles_df.at[v_idx, 'Vehicle Position (x, y)'] = b_task['Task Position (x, y)']
                assigned_tasks.add(b_task['Task ID'])
                
                engagement_details.append({
                    "task_id": b_task['Task ID'], 
                    "task_duration": b_task['Duration (min)'],
                    "travel_time": b_tt, 
                    "engagement_time": b_tot,
                    "normalized_engagement_time": b_tot / b_task['Duration (min)'],
                    "energy_consumed": b_tot
                })
            else:
                # No tasks passed filter: treat as WAIT with penalty
                _agent.update(v_state, action, -2.0, v_state)
        else:
            # ACTION 0: WAIT
            # High penalty if demand is high but agent chooses to idle
            nearby_demand = v_state[1] 
            idle_penalty = -2.5 if nearby_demand == 1 else -0.5
            _agent.update(v_state, 0, idle_penalty, v_state)

    _agent.epsilon = max(_agent.min_epsilon, _agent.epsilon * _agent.epsilon_decay)
    return allocations, engagement_details
