import numpy as np

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class DecentralizedAuctioneer:
    """
    Simulates a decentralized auction process where vehicles bid independently.
    The clearing mechanism is a greedy highest-bid-wins protocol.
    """
    def calculate_local_bid(self, v_row, task, future_tasks, other_idle):
        v_pos = v_row['Vehicle Position (x, y)']
        t_pos = task['Task Position (x, y)']
        dist = calculate_distance(v_pos, t_pos)
        tt = dist / v_row['Speed']
        et = task['Duration (min)'] + tt
        
        if v_row['Battery Level (%)'] < et:
            return -1.0 # Infeasible

        # 1. Proximity Utility
        # Sufficiently high to compete on travel efficiency
        u_dist = 50000.0 / (tt + 0.05)
        
        # 2. Urgency Clearing (Capture Success)
        # Multiplier ensures urgent tasks are cleared quickly
        m_urg = 1.0 + (task['Urgency'] * 15.0)
        
        # 3. Strategic Destination Potential (Chaining)
        radius = 100.0
        t_nearby = sum(1 for p in future_tasks if calculate_distance(t_pos, p) < radius)
        v_nearby = sum(1 for p in other_idle if calculate_distance(t_pos, p) < radius)
        # Stronger pull to productive areas
        m_chain = 1.0 + (2.0 * (t_nearby + 1) / (v_nearby + 1.1))
        
        # 4. Utilization Pressure (Idle-Suppression)
        # Strong linear pull to prevent idleness, with a mild logarithmic boost for longer idle times
        u_idle_val = v_row.get('Idle Time', 0)
        m_idle = 1.0 + (np.log1p(u_idle_val) * 0.8) + (u_idle_val * 0.1)

        return u_dist * m_urg * m_chain * m_idle

def auction_allocation(vehicles_df, tasks_df):
    allocations = {}
    engagement_details = []
    
    idle_v = vehicles_df[~vehicles_df['Busy']].copy()
    if idle_v.empty or tasks_df.empty:
        return allocations, engagement_details

    t_list = tasks_df.to_dict('records')
    v_list = idle_v.to_dict('records')
    
    future_task_pos = [t['Task Position (x, y)'] for t in t_list]
    idle_vehicle_pos = [v['Vehicle Position (x, y)'] for v in v_list]
    
    # Collect all bids in a decentralized manner
    bids = []
    auctioneer = DecentralizedAuctioneer()
    for v_row in v_list:
        for task in t_list:
            bid_val = auctioneer.calculate_local_bid(v_row, task, future_task_pos, idle_vehicle_pos)
            if bid_val > 0:
                bids.append({
                    'v_id': v_row['Vehicle ID'],
                    't_id': task['Task ID'],
                    'bid': bid_val,
                    'task': task,
                    'v_row': v_row
                })
    
    # Clear the auction: Greedy Priority (Highest bid wins)
    # This is NOT a global combinatorial optimizer, but a serial task assignment.
    bids.sort(key=lambda x: x['bid'], reverse=True)
    
    assigned_v = set()
    assigned_t = set()
    
    for b in bids:
        if b['v_id'] in assigned_v or b['t_id'] in assigned_t:
            continue
            
        task = b['task']
        v_row = b['v_row']
        
        dist = calculate_distance(v_row['Vehicle Position (x, y)'], task['Task Position (x, y)'])
        tt = dist / v_row['Speed']
        et = task['Duration (min)'] + tt
        
        real_v_idx = vehicles_df.index[vehicles_df['Vehicle ID'] == v_row['Vehicle ID']][0]
        
        allocations[task['Task ID']] = v_row['Vehicle ID']
        vehicles_df.at[real_v_idx, 'Battery Level (%)'] -= et
        vehicles_df.at[real_v_idx, 'Busy'] = True
        vehicles_df.at[real_v_idx, 'Remaining Duration'] = float(et)
        vehicles_df.at[real_v_idx, 'Vehicle Position (x, y)'] = task['Task Position (x, y)']
        
        assigned_v.add(b['v_id'])
        assigned_t.add(b['t_id'])
        
        engagement_details.append({
            "task_id": task['Task ID'], 
            "task_duration": task['Duration (min)'],
            "travel_time": tt, 
            "engagement_time": et,
            "normalized_engagement_time": et / task['Duration (min)'],
            "energy_consumed": et
        })
        
    return allocations, engagement_details
