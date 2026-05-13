import numpy as np
import random

class TabularQLearner:
    def __init__(self, alpha=0.2, gamma=0.95, epsilon=0.2, epsilon_decay=0.995):
        self.q_table = {}  # Key: (v_state, action_key)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.visit_counts = {}

    def get_vehicle_state(self, vehicle, idle_vehicles):
        """
        Contextual state of the vehicle itself.
        Compact: (Grid_X, Grid_Y, Competition, Idle) -> 3x3x2x2 = 36 states
        """
        pos = vehicle['Vehicle Position (x, y)']
        # 3x3 grid for 500x500 area
        gx = int(pos[0] / 167) 
        gy = int(pos[1] / 167)
        gx = min(2, gx)
        gy = min(2, gy)
        
        # Competition: Binary (Low vs High)
        radius = 120.0
        comp = sum(1 for v_pos in idle_vehicles if np.sqrt((pos[0]-v_pos[0])**2 + (pos[1]-v_pos[1])**2) < radius)
        c_bin = 0 if comp <= 2 else 1
        
        # Idle: Binary (Fresh vs Stale)
        idle_time = vehicle.get('Idle Time', 0)
        i_bin = 0 if idle_time < 20 else 1
        
        return (gx, gy, c_bin, i_bin)

    def get_action_features(self, vehicle, task, future_tasks):
        """
        Features of a specific assignment action.
        Compact: (Dist, Urg, Density) -> 2x2x2 = 8 actions
        """
        v_pos = vehicle['Vehicle Position (x, y)']
        t_pos = task['Task Position (x, y)']
        dist = np.sqrt((v_pos[0] - t_pos[0])**2 + (v_pos[1] - t_pos[1])**2)
        
        # Distance: Close vs Far
        d_bin = 0 if dist < 40 else 1
        
        # Urgency: Normal vs High
        urg = task.get('Urgency', 0)
        u_bin = 0 if urg < 8 else 1
        
        # Density at target: Sparse vs Rich
        radius = 100.0
        density = sum(1 for pos in future_tasks if np.sqrt((t_pos[0]-pos[0])**2 + (t_pos[1]-pos[1])**2) < radius)
        den_bin = 0 if density <= 1 else 1
        
        return (d_bin, u_bin, den_bin)

    def get_q(self, v_state, action_features):
        key = (v_state, action_features)
        if key not in self.q_table:
            # Informed Initialization (Moderate bias)
            d, u, den = action_features
            # Normalized scale: [0, 20]
            self.q_table[key] = 12.0 - (4.0 * d) + (4.0 * u) + (2.0 * den)
        return self.q_table[key]

    def get_max_q_for_state(self, v_state):
        """
        Approximates best value from a vehicle state across potential action types.
        """
        best_q = -float('inf')
        found = False
        for (st, _), val in self.q_table.items():
            if st == v_state:
                if val > best_q:
                    best_q = val
                found = True
        return best_q if found else 5.0 # Default optimism

    def update(self, v_state, action_features, reward, next_v_state):
        key = (v_state, action_features)
        if key not in self.q_table:
            self.get_q(v_state, action_features)
            
        old_val = self.q_table[key]
        
        # High discounting for multi-step chaining
        next_max_q = self.get_max_q_for_state(next_v_state)
        target = reward + 0.98 * next_max_q
        
        # Higher alpha for fast convergence in low-step simulations
        self.q_table[key] = old_val + 0.5 * (target - old_val)
        self.visit_counts[key] = self.visit_counts.get(key, 0) + 1

    def decay_epsilon(self):
        # Faster decay to lock-in learned policy
        self.epsilon = max(0.02, self.epsilon * 0.97)

# Global Agent Singleton
_agent = TabularQLearner()

def qlearning_allocation(vehicles_df, tasks_df, current_time=0):
    global _agent
    allocations = {}
    engagement_details = []
    
    idle_indices = vehicles_df[~vehicles_df['Busy']].index.tolist()
    if not idle_indices or tasks_df.empty:
        return allocations, engagement_details

    task_records = tasks_df.to_dict('records')
    future_pos = [t['Task Position (x, y)'] for t in task_records]
    idle_pos = [vehicles_df.at[i, 'Vehicle Position (x, y)'] for i in idle_indices]

    # Pre-calculate states
    vehicle_states = {}
    for idx in idle_indices:
        vehicle_states[idx] = _agent.get_vehicle_state(vehicles_df.loc[idx], idle_pos)

    # Assignment Bids
    candidate_bids = []
    for v_idx in idle_indices:
        vehicle = vehicles_df.loc[v_idx]
        v_state = vehicle_states[v_idx]
        
        for t_idx, task in enumerate(task_records):
            a_features = _agent.get_action_features(vehicle, task, future_pos)
            
            # Distance/Battery check
            v_p = vehicle['Vehicle Position (x, y)']
            t_p = task['Task Position (x, y)']
            dist = np.sqrt((v_p[0]-t_p[0])**2 + (v_p[1]-t_p[1])**2)
            tt = dist / vehicle['Speed']
            total_time = task['Duration (min)'] + tt
            
            if vehicle['Battery Level (%)'] >= total_time:
                # Epsilon-Greedy Choice
                if random.random() < _agent.epsilon:
                    score = random.uniform(0, 15) # Exploration
                else:
                    score = _agent.get_q(v_state, a_features)
                
                candidate_bids.append({
                    'v_idx': v_idx,
                    't_idx': t_idx,
                    'score': score,
                    'v_state': v_state,
                    'a_features': a_features,
                    'task': task,
                    'tt': tt,
                    'total_time': total_time,
                    'dist': dist
                })

    # Execute Assignments
    candidate_bids.sort(key=lambda x: x['score'], reverse=True)
    assigned_vs = set()
    assigned_ts = set()

    for bid in candidate_bids:
        if bid['v_idx'] in assigned_vs or bid['t_idx'] in assigned_ts:
            continue
            
        v_idx = bid['v_idx']
        task = bid['task']
        vehicle = vehicles_df.loc[v_idx]
        
        # ------------------------------------------------------------------
        # LEARNED REWARD DESIGN (Enhanced)
        # ------------------------------------------------------------------
        # Capture reward (+20.0) -> High base to favor activity over idling
        # Urgency bonus (+1.5 per level)
        # Distance penalty (-0.5 per travel time unit)
        urg_bonus = task['Urgency'] * 1.5
        dist_penalty = bid['tt'] * 0.5
        reward = 20.0 + urg_bonus - dist_penalty
        
        # Define next state (at target position)
        next_v = vehicle.copy()
        next_v['Vehicle Position (x, y)'] = task['Task Position (x, y)']
        next_v['Idle Time'] = 0
        # Use existing idle positions as proxy for future competition
        next_v_state = _agent.get_vehicle_state(next_v, idle_pos)
        
        # TD Update
        _agent.update(bid['v_state'], bid['a_features'], reward, next_v_state)
        
        # Apply Assignment
        allocations[task['Task ID']] = vehicle['Vehicle ID']
        vehicles_df.at[v_idx, 'Battery Level (%)'] -= bid['total_time']
        vehicles_df.at[v_idx, 'Busy'] = True
        vehicles_df.at[v_idx, 'Remaining Duration'] = float(bid['total_time'])
        vehicles_df.at[v_idx, 'Vehicle Position (x, y)'] = task['Task Position (x, y)']
        
        assigned_vs.add(v_idx)
        assigned_ts.add(bid['t_idx'])

        engagement_details.append({
            "task_id": task['Task ID'], 
            "task_duration": task['Duration (min)'],
            "travel_time": bid['tt'], 
            "engagement_time": bid['total_time'],
            "normalized_engagement_time": bid['total_time'] / task['Duration (min)'],
            "energy_consumed": bid['total_time']
        })

    # Explicit WAIT learning for remaining idle vehicles
    for v_idx in idle_indices:
        if v_idx not in assigned_vs:
            v_ref = vehicles_df.loc[v_idx]
            v_state = vehicle_states[v_idx]
            # Wait "action" features: dist=infinite, urg=0, density=0 (Sentinel)
            wait_features = (3, 0, 0)
            
            # Idle penalty
            idle_penalty = -0.5
            
            # Next state: same position, increased idle time
            next_v = v_ref.copy()
            next_v['Idle Time'] = v_ref.get('Idle Time', 0) + 1
            next_v_state = _agent.get_vehicle_state(next_v, idle_pos)
            
            _agent.update(v_state, wait_features, idle_penalty, next_v_state)

    _agent.decay_epsilon()
    return allocations, engagement_details

