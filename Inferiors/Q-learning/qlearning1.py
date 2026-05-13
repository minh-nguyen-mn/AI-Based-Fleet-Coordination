import numpy as np

class TabularQLearner:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Rolling stats for normalization/tracking
        self.visit_counts = {}

    def get_state(self, vehicle, task, future_tasks, idle_vehicles, current_time):
        """
        Simplified State for faster convergence.
        """
        v_pos = vehicle['Vehicle Position (x, y)']
        t_pos = task['Task Position (x, y)']
        dist = np.sqrt((v_pos[0] - t_pos[0])**2 + (v_pos[1] - t_pos[1])**2)
        
        # 1. Proximity (3 bins)
        d_bin = min(int(dist / 20), 2)
        
        # 2. Local Work Availability (2 bins)
        radius = 100.0
        local_tasks = sum(1 for p in future_tasks if np.sqrt((t_pos[0]-p[0])**2 + (t_pos[1]-p[1])**2) < radius)
        dense_bin = 1 if local_tasks > 1 else 0
        
        # 3. Idle Pressure (2 bins)
        idle_bin = 1 if vehicle.get('Idle Time', 0) > 15 else 0
        
        return (d_bin, dense_bin, idle_bin)

    def get_q(self, state):
        if state not in self.q_table:
            d, den, i = state
            # Optimistic: prefers dense areas and proximity
            self.q_table[state] = 100000.0 - (10000.0 * d) + (20000.0 * den) + (5000.0 * i)
        return self.q_table[state]

    def update(self, state, reward, next_state_max_q=0):
        old_val = self.get_q(state)
        # Higher lookahead for chaining
        target = reward + 0.9 * next_state_max_q
        self.q_table[state] = old_val + 0.2 * (target - old_val)
        self.visit_counts[state] = self.visit_counts.get(state, 0) + 1

# Global Agent Singleton
_agent = TabularQLearner()

def qlearning_allocation(vehicles_df, tasks_df, current_time=0):
    global _agent
    allocations = {}
    engagement_details = []
    
    idle_v_indices = vehicles_df[~vehicles_df['Busy']].index.tolist()
    if not idle_v_indices or tasks_df.empty:
        return allocations, engagement_details

    task_records = tasks_df.to_dict('records')
    future_task_pos = [t['Task Position (x, y)'] for t in task_records]
    idle_vehicle_pos = [vehicles_df.at[i, 'Vehicle Position (x, y)'] for i in idle_v_indices]

    # Distributed Bidding
    bids = []
    for v_idx in idle_v_indices:
        vehicle = vehicles_df.loc[v_idx]
        for t_idx, task in enumerate(task_records):
            state = _agent.get_state(vehicle, task, future_task_pos, idle_vehicle_pos, current_time)
            q_val = _agent.get_q(state)
            
            # Distance and Battery feasibility check
            dist = np.sqrt((vehicle['Vehicle Position (x, y)'][0]-task['Task Position (x, y)'][0])**2 + 
                           (vehicle['Vehicle Position (x, y)'][1]-task['Task Position (x, y)'][1])**2)
            tt = dist / vehicle['Speed']
            et = task['Duration (min)'] + tt
            
            if vehicle['Battery Level (%)'] >= et:
                bids.append({
                    'v_idx': v_idx, 
                    't_idx': t_idx, 
                    'q': q_val, 
                    'state': state, 
                    'task': task,
                    'et': et,
                    'tt': tt
                })

    # Clear assignments by Q-Value (Preference)
    bids.sort(key=lambda x: x['q'], reverse=True)
    assigned_v = set()
    assigned_t = set()

    for b in bids:
        if b['v_idx'] in assigned_v or b['t_idx'] in assigned_t:
            continue
            
        v_idx = b['v_idx']
        task = b['task']
        vehicle = vehicles_df.loc[v_idx]
        
        # ------------------------------------------------------------------
        # LONG-HORIZON REWARD: Busy-Time Dominance & High Capture
        # ------------------------------------------------------------------
        # 1. Total Occupancy reward (Forces the agent to stay busy like Greedy)
        r_busy = 7000.0 * b['et']
        
        # 2. Travel Penalty (Balanced for energy efficiency dominance)
        r_eff = -5000.0 * b['tt']
        
        # 3. Massive Capture Bonus (num_tasks driver)
        r_capture = 800000.0 + (200000.0 * task['Urgency'])
        
        # 4. Strategic Destination (Chaining)
        r_chain = 100000.0 * b['state'][1] # dense_bin
        
        reward = r_busy + r_eff + r_capture + r_chain
        
        # Stability: bootstrap from dense areas
        next_max_q = 200000.0 if b['state'][1] > 0 else 80000.0
        _agent.update(b['state'], reward, next_max_q)
        
        # Execute assignment
        allocations[task['Task ID']] = vehicle['Vehicle ID']
        vehicles_df.at[v_idx, 'Battery Level (%)'] -= b['et']
        vehicles_df.at[v_idx, 'Busy'] = True
        vehicles_df.at[v_idx, 'Remaining Duration'] = float(b['et'])
        vehicles_df.at[v_idx, 'Vehicle Position (x, y)'] = task['Task Position (x, y)']
        
        assigned_v.add(v_idx)
        assigned_t.add(b['t_idx'])

        engagement_details.append({
            "task_id": task['Task ID'], 
            "task_duration": task['Duration (min)'],
            "travel_time": b['tt'], 
            "engagement_time": b['et'],
            "normalized_engagement_time": b['et'] / task['Duration (min)'],
            "energy_consumed": b['et']
        })

    # Learning for idle vehicles (Explicit penalty for staying idle)
    for v_idx in idle_v_indices:
        if v_idx not in assigned_v:
            curr_v = vehicles_df.loc[v_idx]
            # State: (No task distance, No local density, Current Idle Bin)
            # Using a sentinel 'no-task' action proxy (0, 0, idle_bin)
            idle_state = (0, 0, 1 if curr_v.get('Idle Time', 0) > 15 else 0)
            _agent.update(idle_state, -50000.0) 

    return allocations, engagement_details

