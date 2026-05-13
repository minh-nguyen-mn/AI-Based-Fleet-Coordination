import numpy as np
import random

class TabularQLearner:
    def __init__(self, alpha=0.6, gamma=0.8, epsilon=0.15):
        self.q_table = {}  # (state, action_id)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.98
        self.min_epsilon = 0.02
        
    def get_state(self, v_row, tasks, idle_vehicles):
        """
        Decision-centric state: (Location_Grid, Local_Competition, Task_Opportunity)
        500x500 -> 3x3 grid (9)
        Competition: Binary (Many idle nearby?)
        Opportunity: Binary (Many tasks nearby?)
        Total: 9 * 2 * 2 = 36 states.
        """
        pos = v_row['Vehicle Position (x, y)']
        gx, gy = int(pos[0]/167), int(pos[1]/167)
        gx, gy = min(2, gx), min(2, gy)
        loc = gx * 3 + gy
        
        radius = 150.0
        # Competition: idle vehicles (excluding self)
        comp = sum(1 for p in idle_vehicles if np.sqrt((pos[0]-p[0])**2 + (pos[1]-p[1])**2) < radius) - 1
        comp_bin = 1 if comp >= 2 else 0
        
        # Opportunity: available tasks
        opp = sum(1 for t in tasks if np.sqrt((pos[0]-t['Task Position (x, y)'][0])**2 + (pos[1]-t['Task Position (x, y)'][1])**2) < radius)
        opp_bin = 1 if opp >= 2 else 0
        
        return (loc, comp_bin, opp_bin)

    def get_q(self, state, action):
        if (state, action) not in self.q_table:
            # Start at 0, no heuristic bias to ensure learning defines behavior
            self.q_table[(state, action)] = 0.0
        return self.q_table[(state, action)]

    def update(self, state, action, reward, next_state):
        current_q = self.get_q(state, action)
        
        # Get max Q for next state
        next_max_q = max([self.get_q(next_state, a) for a in range(4)])
        
        # Bellman equation
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
    
    # Process vehicles sequentially to maintain decentralized feel
    random.shuffle(idle_indices)
    
    assigned_tasks = set()
    
    for v_idx in idle_indices:
        vehicle = vehicles_df.loc[v_idx]
        v_state = _agent.get_state(vehicle, tasks, idle_pos)
        
        # Action Selection (Archetypes)
        # 0: WAIT, 1: FOCUS_URGENCY, 2: FOCUS_PROXIMITY, 3: FOCUS_STRATEGIC (Density)
        # Note: action 1 (URGENCY) now scans more broadly to maximize throughput.
        if random.random() < _agent.epsilon:
            action = random.randint(0, 3)
        else:
            qs = [_agent.get_q(v_state, a) for a in range(4)]
            action = np.argmax(qs)
            
        # Target Task Discovery per Archetype
        target_bid = None
        
        # Only evaluate tasks that aren't assigned yet
        available_tasks = [t for t in tasks if t['Task ID'] not in assigned_tasks]
        
        if action > 0 and available_tasks:
            scored_tasks = []
            for task in available_tasks:
                v_p = vehicle['Vehicle Position (x, y)']
                t_p = task['Task Position (x, y)']
                dist = np.sqrt((v_p[0]-t_p[0])**2 + (v_p[1]-t_p[1])**2)
                tt = dist / vehicle['Speed']
                total_t = task['Duration (min)'] + tt
                
                if vehicle['Battery Level (%)'] < total_t:
                    continue
                    
                # Scorer based on archetype
                if action == 1: # GLOBAL URGENCY
                    score = task['Urgency'] * 5.0 - (tt * 0.01)
                elif action == 2: # PROXIMITY
                    score = 250.0 / (tt + 0.1)
                else: # STRATEGIC
                    radius = 200.0
                    dens = sum(1 for t2 in tasks if np.sqrt((t_p[0]-t2['Task Position (x, y)'][0])**2 + (t_p[1]-t2['Task Position (x, y)'][1])**2) < radius)
                    score = dens * 10.0 - (tt * 0.05)
                
                scored_tasks.append((score, task, tt, total_t))
            
            if scored_tasks:
                scored_tasks.sort(key=lambda x: x[0], reverse=True)
                _, best_task, best_tt, best_total = scored_tasks[0]
                
                # REWARD: Thru-max reward
                r_thru = 12.0 # Extreme base reward to minimize idle
                r_urg = (best_task['Urgency'] / 10.0) * 4.0
                r_cost = -(best_tt / 80.0) # Low penalty to prioritize capture
                reward = r_thru + r_urg + r_cost
                
                # Next State: after task
                next_v = vehicle.copy()
                next_v['Vehicle Position (x, y)'] = best_task['Task Position (x, y)']
                next_state = _agent.get_state(next_v, tasks, idle_pos)
                
                # Update
                _agent.update(v_state, action, reward, next_state)
                
                # Execute
                allocations[best_task['Task ID']] = vehicle['Vehicle ID']
                vehicles_df.at[v_idx, 'Battery Level (%)'] -= best_total
                vehicles_df.at[v_idx, 'Busy'] = True
                vehicles_df.at[v_idx, 'Remaining Duration'] = float(best_total)
                vehicles_df.at[v_idx, 'Vehicle Position (x, y)'] = best_task['Task Position (x, y)']
                
                assigned_tasks.add(best_task['Task ID'])
                engagement_details.append({
                    "task_id": best_task['Task ID'], 
                    "task_duration": best_task['Duration (min)'],
                    "travel_time": best_tt, 
                    "engagement_time": best_total,
                    "normalized_engagement_time": best_total / best_task['Duration (min)'],
                    "energy_consumed": best_total
                })
            else:
                # Forced WAIT if no feasible tasks
                _agent.update(v_state, action, -5.0, v_state)
        else:
            # ACTION 0: WAIT (Extreme penalty to force task seeking)
            _agent.update(v_state, 0, -3.0, v_state)

    _agent.epsilon = max(_agent.min_epsilon, _agent.epsilon * _agent.epsilon_decay)
    return allocations, engagement_details

