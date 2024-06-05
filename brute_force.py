import math
from tqdm import tqdm
import numpy as np

def highest_rewards_for_end_nodes(start_node, x, adjacency_matrix):

    
    def dfs(node, end_node, current_path):
        nonlocal highest_rewards, paths_with_highest_rewards
        current_path.append(node)

        if node == end_node:
            current_reward = 0

            for node_idx in range(1, len(current_path)):
                current_reward += math.log(x[current_path[node_idx]] + 1)
                #current_reward = x[node_num].sum()
                #current_reward = accumulated_w[node_num]

            if current_reward > highest_rewards[end_node]:
                highest_rewards[node] = current_reward
                paths_with_highest_rewards[node] = current_path.copy()

            return

        for neighbor in range(len(adjacency_matrix[node])):
            if adjacency_matrix[node][neighbor] == 1 and neighbor not in current_path:
                old_path = current_path.copy()
                dfs(neighbor, end_node, old_path)

    highest_rewards = np.zeros(len(adjacency_matrix))
    paths_with_highest_rewards = [[] for _ in range(len(adjacency_matrix))]

    for end_node in tqdm(range(len(adjacency_matrix))):
        dfs(start_node, end_node, [])
    
    return highest_rewards, paths_with_highest_rewards