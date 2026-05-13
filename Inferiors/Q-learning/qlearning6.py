import numpy as np
import random

class PolicyGradientAllocator:
    def __init__(self, learning_rate=0.01):
        # Intercept, dist, urg, dens, batt
        self.weights = np.array([3.0, -0.5, 1.0, 0.3, 0.1])
        self.lr = learning_rate
        self.replay_buffer = []
        self.max_buffer = 200

    def get_features(self, vehicle, task, future_tasks):
        v_pos = vehicle['Vehicle Position (x, y)']
        t_pos = task['Task Position (x, y)']
        dist = np.sqrt((v_pos[0]-t_pos[0])**2 + (v_pos[1]-t_pos[1])**2)
        tt = dist / vehicle['Speed']
        
        f_dist = tt / 15.0
        f_urg = task['Urgency'] / 10.0
        radius = 120.0
        density = sum(1 for t in future_tasks if np.sqrt((t_pos[0]-t[0])**2 + (t_pos[1]-t[1])**2) < radius)
        f_dens = density / 4.0
        f_batt = vehicle['Battery Level (%)'] / 100.0
        
        return np.array([1.0, f_dist, f_urg, f_dens, f_batt])

    def score(self, features):
        return np.dot(self.weights, features)

    def update_policy(self, round_data):
        if not round_data:
            return
        
        # Experience Replay: Add this round to buffer
        self.replay_buffer.extend(round_data)
        if len(self.replay_buffer) > self.max_buffer:
            self.replay_buffer = self.replay_buffer[-self.max_buffer:]
            
        # Sample mini-batch from buffer for stability
        batch_size = min(len(self.replay_buffer), 16)
        batch = random.sample(self.replay_buffer, batch_size)
        
        rewards = [d[1] for d in batch]
        mean_reward = np.mean(rewards)
        
        for features, reward in batch:
            advantage = reward - mean_reward
            self.weights += self.lr * advantage * features
            
        self.weights = np.clip(self.weights, -10, 10)

# Global Singleton Agent
_agent = PolicyGradientAllocator()

def qlearning_allocation(vehicles_df, tasks_df, current_time=0):
    global _agent
    allocations = {}
    engagement_details = []
    
    idle_indices = vehicles_df[~vehicles_df['Busy']].index.tolist()
    if not idle_indices or tasks_df.empty:
        return allocations, engagement_details

    tasks = tasks_df.to_dict('records')
    future_pos = [t['Task Position (x, y)'] for t in tasks]
    
    # Context-Aware Matching
    round_updates = []
    assigned_tasks = set()
    
    # Process vehicles in a random order to ensure decentralized flavor
    random.shuffle(idle_indices)
    
    for v_idx in idle_indices:
        vehicle = vehicles_df.loc[v_idx]
        available_tasks = [t for t in tasks if t['Task ID'] not in assigned_tasks]
        if not available_tasks:
            break
            
        # 1. Feasibility & Feature Extraction
        candidates = []
        for task in available_tasks:
            v_p = vehicle['Vehicle Position (x, y)']
            t_p = task['Task Position (x, y)']
            dist = np.sqrt((v_p[0]-t_p[0])**2 + (v_p[1]-t_p[1])**2)
            tt = dist / vehicle['Speed']
            tot = task['Duration (min)'] + tt
            
            if vehicle['Battery Level (%)'] >= tot:
                feats = _agent.get_features(vehicle, task, future_pos)
                score = _agent.score(feats)
                candidates.append({
                    'task': task,
                    'feats': feats,
                    'score': score,
                    'tt': tt,
                    'tot': tot
                })
        
        if not candidates:
            continue
            
        # 2. Softmax Selection with Temperature T=0.5 (Greedier)
        temp = 0.5
        scores = np.array([c['score'] for c in candidates])
        exp_s = np.exp((scores - np.max(scores)) / temp)
        probs = exp_s / exp_s.sum()
        
        idx = np.random.choice(len(candidates), p=probs)
        winner = candidates[idx]
        
        # 3. Learning Reward Calculation
        # Productivity = Work / (Work + Travel)
        efficiency = winner['task']['Duration (min)'] / winner['tot']
        urgency_factor = (winner['task']['Urgency'] / 10.0)
        
        # Utilization Reward: punish deadhead travel that takes too long
        reward = efficiency + urgency_factor
        
        round_updates.append((winner['feats'], reward))
        
        # 4. Execute Assignment
        best_task = winner['task']
        allocations[best_task['Task ID']] = vehicle['Vehicle ID']
        vehicles_df.at[v_idx, 'Battery Level (%)'] -= winner['tot']
        vehicles_df.at[v_idx, 'Busy'] = True
        vehicles_df.at[v_idx, 'Remaining Duration'] = float(winner['tot'])
        vehicles_df.at[v_idx, 'Vehicle Position (x, y)'] = best_task['Task Position (x, y)']
        
        assigned_tasks.add(best_task['Task ID'])
        engagement_details.append({
            "task_id": best_task['Task ID'], 
            "task_duration": best_task['Duration (min)'],
            "travel_time": winner['tt'], 
            "engagement_time": winner['tot'],
            "normalized_engagement_time": winner['tot'] / best_task['Duration (min)'],
            "energy_consumed": winner['tot']
        })

    # Batch weight update based on this round's successes/failures
    _agent.update_policy(round_updates)
    
    return allocations, engagement_details
