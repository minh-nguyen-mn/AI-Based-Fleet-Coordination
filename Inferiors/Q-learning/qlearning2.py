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
        Policy-driven state: balances immediate urgency, travel effort, and future utility.
        States: (Dist_Bin, Urg_Bin, Density_Bin)
        """
        v_pos = vehicle['Vehicle Position (x, y)']
        t_pos = task['Task Position (x, y)']
        dist = np.sqrt((v_pos[0] - t_pos[0])**2 + (v_pos[1] - t_pos[1])**2)
        
        # 1. Distance (3 bins for finer travel cost awareness)
        if dist < 20: d_bin = 0
        elif dist < 50: d_bin = 1
        else: d_bin = 2
        
        # 2. Urgency (3 bins: critically old tasks need dominance)
        urg = task.get('Urgency', 0)
        if urg < 4: u_bin = 0
        elif urg < 10: u_bin = 1
        else: u_bin = 2
        
        # 3. Strategic Destination Potential (Density)
        radius = 110.0
        future_density = sum(1 for p in future_tasks if np.sqrt((t_pos[0]-p[0])**2 + (t_pos[1]-p[1])**2) < radius)
        dense_bin = 1 if future_density > 1 else 0
        
        return (d_bin, u_bin, dense_bin)

    def get_q(self, state):
        if state not in self.q_table:
            d, u, den = state
            # Strategic Bias: Favor Urgency, penalize Distance, value Density.
            # Initial values calibrated to encourage exploration.
            base = 1500000.0
            self.q_table[state] = base + (300000.0 * u) - (200000.0 * d) + (150000.0 * den)
        return self.q_table[state]

    def update(self, state, reward, next_state_max_q=0):
        if state not in self.q_table:
            self.get_q(state)
        old_val = self.q_table[state]
        # Strong discounting for multi-step chaining (gamma=0.9)
        target = reward + 0.9 * next_state_max_q
        # Moderate alpha for stable convergence
        self.q_table[state] = old_val + 0.3 * (target - old_val)
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
                # Add a fine-grained tie-breaker to the Q-value to distinguish within bins
                # (Urgency + distance weight is minor compared to Q, but prevents random noise)
                tie_breaker = (task['Urgency'] * 1000.0) - (dist * 100.0)
                bids.append({
                    'v_idx': v_idx, 
                    't_idx': t_idx, 
                    'q': q_val + tie_breaker, 
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
        # STRATEGIC REWARD: High-Value Chaining
        # ------------------------------------------------------------------
        # 1. Capture Reward (Primary driver for num_tasks)
        # Scaled significantly to dwarf minor travel noise.
        r_capture = 2500000.0 * (1.0 + task['Urgency'] * 0.15)
        
        # 2. Travel Penalty (Non-linear to strongly discourage long repositioning)
        r_travel = -20000.0 * (b['tt'] ** 1.1)
        
        # 3. Productive Engagement (Reward staying busy)
        r_prod = 10000.0 * b['et']
        
        # 4. Successor Potential (Density at target)
        r_chained_potential = 700000.0 * b['state'][2] # dense_bin
        
        reward = r_capture + r_travel + r_prod + r_chained_potential
        
        # Next Max Q - Optimistic lookahead based on density
        next_val = 1800000.0 if b['state'][2] > 0 else 600000.0
        _agent.update(b['state'], reward, next_val)
        
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

    # Global Idle Penalty: Aggressive discouragement of passivity
    for v_idx in idle_v_indices:
        if v_idx not in assigned_v:
            curr_v = vehicles_df.loc[v_idx]
            # (Far, Low Urgency, Poor Density)
            idle_state = (2, 0, 0)
            _agent.update(idle_state, -500000.0) 

    return allocations, engagement_details

