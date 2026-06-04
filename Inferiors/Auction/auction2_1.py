import numpy as np

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class DecentralizedAuctioneer:
    """
    Fundamentally decentralized bidding mechanism.
    Each vehicle computes its bid based on local utility and strategic heuristics.
    The task clearing is a greedy highest-bid-wins protocol.
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
        # Base bid on travel effort.
        u_base = 50000.0 / (tt + 1.0)
        
        # 2. Urgency Clearing
        # Non-linear boost to ensure prompt capture of aging tasks.
        m_urg = 1.0 + (task['Urgency'] * 2.0)
        
        # 3. Strategic Spatial Coordination (Multi-step lookahead)
        # Regional Density Estimate: How many future tasks are near the target?
        radius = 25.0
        nearby_tasks = sum(1 for p in future_tasks if calculate_distance(t_pos, p) < radius)
        
        # Coverage Factor: How many other idle vehicles are near the target?
        # Increased penalty to 1.5 to improve regional coverage balance.
        nearby_idle = sum(1 for p in other_idle if calculate_distance(t_pos, p) < radius)
        m_coordination = (1.0 + (nearby_tasks * 10.0)) / (1.0 + (nearby_idle * 1.5))
        
        # 4. Global Load Balancing
        # Increased aggression multiplier to force idle vehicles into work.
        u_idle_val = v_row.get('Idle Time', 0)
        m_aggression = 1.0 + (u_idle_val * 2.0) + (u_idle_val**2 * 0.1)
        
        # 5. Resource Efficient Sourcing
        # Prefer assignments that don't leave the vehicle in a "dead zone".
        # Value = (Base * Urgency * Coordination * Aggression)
        bid = u_base * m_urg * m_coordination * m_aggression
        
        return bid

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
    
    # Independent Bidding Phase
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
    
    # Clearing Phase: Highest bid wins (Decentralized Coordination)
    # This remains a serial allocator, not a global combinatorial solver.
    bids.sort(key=lambda x: x['bid'], reverse=True)
    
    assigned_v = set()
    assigned_t = set()
    
    for b in bids:
        if b['v_id'] in assigned_v or b['t_id'] in assigned_t:
            continue
            
        task = b['task']
        v_row = b['v_row']
        
        v_pos = v_row['Vehicle Position (x, y)']
        t_pos = task['Task Position (x, y)']
        dist = calculate_distance(v_pos, t_pos)
        tt = dist / v_row['Speed']
        et = task['Duration (min)'] + tt
        
        # Check mapping to original DF
        real_v_idx = vehicles_df.index[vehicles_df['Vehicle ID'] == v_row['Vehicle ID']][0]
        
        # Assignment
        allocations[task['Task ID']] = v_row['Vehicle ID']
        vehicles_df.at[real_v_idx, 'Battery Level (%)'] -= et
        vehicles_df.at[real_v_idx, 'Busy'] = True
        vehicles_df.at[real_v_idx, 'Remaining Duration'] = float(et)
        vehicles_df.at[real_v_idx, 'Vehicle Position (x, y)'] = t_pos
        
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
