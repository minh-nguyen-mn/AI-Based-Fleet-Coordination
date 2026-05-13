import numpy as np

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class LearningEnhancedAuctioneer:
    """
    Architectural Redesign: Learning-Enhanced Decentralized Auctioneer.
    Combines the robust conflict resolution of Auctions with a learned 
    'Regional Value Map' to predict future demand (Temporal Difference Learning).
    """
    def __init__(self):
        # 3x3 Grid of values representing "Future Opportunity" in a region
        self.grid_size = 500 / 3.0
        self.value_map = np.full((3, 3), 1.0)
        self.learning_rate = 0.1
        self.gamma = 0.8
        
    def get_grid_pos(self, pos):
        gx = int(min(2, max(0, pos[0] // self.grid_size)))
        gy = int(min(2, max(0, pos[1] // self.grid_size)))
        return gx, gy

    def calculate_local_bid(self, v_row, tasks, other_idle):
        v_pos = v_row['Vehicle Position (x, y)']
        v_results = []
        
        for task in tasks:
            t_pos = task['Task Position (x, y)']
            dist = calculate_distance(v_pos, t_pos)
            tt = dist / v_row['Speed']
            total_time = task['Duration (min)'] + tt
            
            if v_row['Battery Level (%)'] < total_time:
                continue

            # 1. Base Efficiency
            efficiency = task['Duration (min)'] / (total_time + 0.5)
            
            # 2. Regional 'Future' Value (Learned Component)
            gx, gy = self.get_grid_pos(t_pos)
            future_value = self.value_map[gx, gy]
            
            # 3. Urgency Bias (Exponential)
            urgency_m = np.exp(task['Urgency'] / 10.0) 
            
            # 4. Supply-Demand Balance (Crowding)
            radius = 120.0
            nearby_idle = sum(1 for p in other_idle if calculate_distance(t_pos, p) < radius)
            congestion_penalty = 1.0 / (1.0 + nearby_idle)
            
            # Combined Utility: Immediate Reward + Discounted Future Region Value
            utility = (efficiency * urgency_m * congestion_penalty) + (0.3 * future_value)
            
            v_results.append((utility, task, tt, total_time, gx, gy))
            
        return sorted(v_results, key=lambda x: x[0], reverse=True)

    def update_values(self, task_region, reward):
        """
        TD Update for the regional map.
        When a task is completed, we update the value of that region.
        """
        gx, gy = task_region
        # Temporal Difference: Value(s) = Value(s) + alpha * (Reward + gamma * MaxValue - Value(s))
        # Here reward is the task duration (productivity signal)
        current_v = self.value_map[gx, gy]
        self.value_map[gx, gy] = current_v + self.learning_rate * (reward - current_v)

# Global persistent learner
_le_auctioneer = LearningEnhancedAuctioneer()

def auction_allocation(vehicles_df, tasks_df):
    global _le_auctioneer
    allocations = {}
    engagement_details = []
    
    idle_v = vehicles_df[~vehicles_df['Busy']].copy()
    if idle_v.empty or tasks_df.empty:
        return allocations, engagement_details

    t_list = tasks_df.to_dict('records')
    v_list = idle_v.to_dict('records')
    idle_vehicle_pos = [v['Vehicle Position (x, y)'] for v in v_list]
    
    v_options = {v['Vehicle ID']: _le_auctioneer.calculate_local_bid(v, t_list, idle_vehicle_pos) for v in v_list}

    assigned_v = set()
    assigned_t = set()
    
    # Priority Ranking (Bidding on Relative Advantage)
    while len(assigned_v) < len(v_list) and len(assigned_t) < len(t_list):
        candidates = []
        for v_id, options in v_options.items():
            if v_id in assigned_v: continue
            valid = [o for o in options if o[1]['Task ID'] not in assigned_t]
            if not valid: continue
            
            # Margin Bid = Advantage over next best
            best_u, task_ref, tt, tot, gx, gy = valid[0]
            if len(valid) > 1:
                margin = (best_u - valid[1][0]) / (best_u + 1.0)
                bid = (best_u * 0.4) + (margin * 10.0)
            else:
                bid = best_u
                
            candidates.append({'v_id': v_id, 'bid': bid, 'task': task_ref, 'tt': tt, 'total': tot, 'region': (gx, gy)})
            
        if not candidates: break
        
        # Winner selection
        winner = max(candidates, key=lambda x: x['bid'])
        v_idx = vehicles_df.index[vehicles_df['Vehicle ID'] == winner['v_id']][0]
        t_id = winner['task']['Task ID']
        
        # Immediate Value Update (Reward Shaping)
        productivity_signal = winner['task']['Duration (min)'] / winner['total']
        _le_auctioneer.update_values(winner['region'], productivity_signal)
        
        # Execute
        allocations[t_id] = winner['v_id']
        vehicles_df.at[v_idx, 'Battery Level (%)'] -= winner['total']
        vehicles_df.at[v_idx, 'Busy'] = True
        vehicles_df.at[v_idx, 'Remaining Duration'] = float(winner['total'])
        vehicles_df.at[v_idx, 'Vehicle Position (x, y)'] = winner['task']['Task Position (x, y)']
        
        assigned_v.add(winner['v_id'])
        assigned_t.add(t_id)
        
        engagement_details.append({
            "task_id": t_id, 
            "task_duration": winner['task']['Duration (min)'],
            "travel_time": winner['tt'], 
            "engagement_time": winner['total'],
            "normalized_engagement_time": winner['total'] / winner['task']['Duration (min)'],
            "energy_consumed": winner['total']
        })
    
    return allocations, engagement_details
