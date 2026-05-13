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

            # 1. Proximity Utility
            u_prox = 200.0 / (tt + 1.0)
            
            # 2. Urgency (Critical for throughput)
            u_urg = task['Urgency'] * 40.0
            
            # 3. Future Accessibility
            u_acc = 0
            radius = 250.0 
            for t2 in tasks:
                if calculate_distance(t_pos, t2['Task Position (x, y)']) < radius:
                    u_acc += 15.0
            
            # 4. Aggressive Capture (Idle time boost)
            u_idle = v_row.get('Idle Time', 0) * 15.0
            
            # 5. Energy Margin (Allow long travel if battery is high)
            u_energy = (v_row['Battery Level (%)'] - total_time) * 2.0
            
            utility = u_prox + u_urg + u_acc + u_idle + u_energy
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
    v_bids = {v['Vehicle ID']: auctioneer.calculate_local_bid(v, t_list, idle_vehicle_pos) for v in v_list}

    assigned_v = set()
    assigned_t = set()
    
    while len(assigned_v) < len(v_list) and len(assigned_t) < len(t_list):
        best_overall = None
        
        for v_id, tasks_for_v in v_bids.items():
            if v_id in assigned_v: continue
            
            valid_tasks = [item for item in tasks_for_v if item[1]['Task ID'] not in assigned_t]
            if not valid_tasks: continue
            
            best_utility, task, tt, total_time = valid_tasks[0]
            
            # Focused throughput: Priority given to high absolute utility
            # to ensure we don't leave tasks unassigned in low-competition rounds.
            score = best_utility
                
            if best_overall is None or score > best_overall['score']:
                best_overall = {
                    'v_id': v_id,
                    'score': score,
                    't_id': task['Task ID'],
                    'task': task,
                    'tt': tt,
                    'total_time': total_time
                }
        
        if best_overall is None: break
        
        v_id = best_overall['v_id']
        t_id = best_overall['t_id']
        task = best_overall['task']
        real_v_idx = vehicles_df.index[vehicles_df['Vehicle ID'] == v_id][0]
        
        allocations[t_id] = v_id
        vehicles_df.at[real_v_idx, 'Battery Level (%)'] -= best_overall['total_time']
        vehicles_df.at[real_v_idx, 'Busy'] = True
        vehicles_df.at[real_v_idx, 'Remaining Duration'] = float(best_overall['total_time'])
        vehicles_df.at[real_v_idx, 'Vehicle Position (x, y)'] = task['Task Position (x, y)']
        
        assigned_v.add(v_id)
        assigned_t.add(t_id)
        engagement_details.append({
            "task_id": t_id, 
            "task_duration": task['Duration (min)'],
            "travel_time": best_overall['tt'], 
            "engagement_time": best_overall['total_time'],
            "normalized_engagement_time": best_overall['total_time'] / task['Duration (min)'],
            "energy_consumed": best_overall['total_time']
        })
    
    return allocations, engagement_details
