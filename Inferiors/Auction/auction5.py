import numpy as np

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class DecentralizedAuctioneer:
    """
    Fundamentally decentralized bidding mechanism.
    Each vehicle computes its bid based on local utility and strategic heuristics.
    The task clearing is a greedy highest-bid-wins protocol.
    """
    def calculate_local_bid(self, v_row, tasks, other_idle):
        """
        Computes a list of utilities for all feasible tasks.
        Utility = Efficiency * Urgency * (Supply/Demand Balance)
        """
        v_pos = v_row['Vehicle Position (x, y)']
        v_results = []
        
        for task in tasks:
            t_pos = task['Task Position (x, y)']
            dist = calculate_distance(v_pos, t_pos)
            tt = dist / v_row['Speed']
            total_time = task['Duration (min)'] + tt
            
            if v_row['Battery Level (%)'] < total_time:
                continue

            # 1. Efficiency Ratio (Work/Energy)
            # This is the core productivity metric
            efficiency = task['Duration (min)'] / (total_time + 1.0)
            
            # 2. Urgency Multiplier
            urgency_m = 1.0 + (task['Urgency'] * 0.12) # Slightly higher urgency bias
            
            # 3. Supply-Demand Balance (Anti-Herding)
            # We want to avoid herding towards the same task or region.
            radius = 130.0
            nearby_tasks = sum(1 for t in tasks if calculate_distance(t_pos, t['Task Position (x, y)']) < radius)
            nearby_idle = sum(1 for p in other_idle if calculate_distance(t_pos, p) < radius)
            
            # Regional Demand Factor: Favor moving towards high demand, low supply zones
            demand_factor = (1.5 + nearby_tasks) / (1.0 + nearby_idle)
            
            # 4. Continuity Bias
            # Prefer tasks that are longer to keep the vehicle engaged, but don't over-commit
            continuity = np.sqrt(task['Duration (min)'] / 20.0) 
            
            # 5. Strategic Idle Pressure (Increased to eliminate lingering idle time)
            idle_bias = 1.0 + (v_row.get('Idle Time', 0) * 0.05)
            
            utility = efficiency * urgency_m * demand_factor * idle_bias * continuity
            v_results.append((utility, task, tt, total_time))
            
        return sorted(v_results, key=lambda x: x[0], reverse=True)

def auction_allocation(vehicles_df, tasks_df):
    allocations = {}
    engagement_details = []
    
    idle_v = vehicles_df[~vehicles_df['Busy']].copy()
    if idle_v.empty or tasks_df.empty:
        return allocations, engagement_details

    t_list = tasks_df.to_dict('records')
    v_list = idle_v.to_dict('records')
    idle_vehicle_pos = [v['Vehicle Position (x, y)'] for v in v_list]
    
    auctioneer = DecentralizedAuctioneer()
    v_options = {v['Vehicle ID']: auctioneer.calculate_local_bid(v, t_list, idle_vehicle_pos) for v in v_list}

    assigned_v = set()
    assigned_t = set()
    
    # Competitive Priority Clearing
    while len(assigned_v) < len(v_list) and len(assigned_t) < len(t_list):
        candidates = []
        
        for v_id, options in v_options.items():
            if v_id in assigned_v: continue
            
            # Filter remaining valid individual tasks
            valid = [o for o in options if o[1]['Task ID'] not in assigned_t]
            if not valid: continue
            
            best_u, task_ref, tt, tot = valid[0]
            
            # Margin Bidding: Bid = how much better this task is than the next best one
            # Prevents taking a task that another vehicle is much better suited for 
            # if we have a decent fallback.
            if len(valid) > 1:
                margin = best_u - valid[1][0]
            else:
                margin = best_u # No fallback
                
            # Final Bid incorporates both absolute utility and relative advantage
            final_bid = margin + (best_u * 0.2)
            candidates.append({
                'v_id': v_id,
                'final_bid': final_bid,
                'task': task_ref,
                'tt': tt,
                'total_time': tot
            })
            
        if not candidates: break
        
        # Best candidate wins this round
        winner = max(candidates, key=lambda x: x['final_bid'])
        
        real_v_idx = vehicles_df.index[vehicles_df['Vehicle ID'] == winner['v_id']][0]
        t_id = winner['task']['Task ID']
        
        allocations[t_id] = winner['v_id']
        vehicles_df.at[real_v_idx, 'Battery Level (%)'] -= winner['total_time']
        vehicles_df.at[real_v_idx, 'Busy'] = True
        vehicles_df.at[real_v_idx, 'Remaining Duration'] = float(winner['total_time'])
        vehicles_df.at[real_v_idx, 'Vehicle Position (x, y)'] = winner['task']['Task Position (x, y)']
        
        assigned_v.add(winner['v_id'])
        assigned_t.add(t_id)
        
        engagement_details.append({
            "task_id": t_id, 
            "task_duration": winner['task']['Duration (min)'],
            "travel_time": winner['tt'], 
            "engagement_time": winner['total_time'],
            "normalized_engagement_time": winner['total_time'] / winner['task']['Duration (min)'],
            "energy_consumed": winner['total_time']
        })
    
    return allocations, engagement_details
