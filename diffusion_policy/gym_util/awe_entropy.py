import numpy as np
import copy
import torch
import os


device = torch.device("cpu")
""" DP waypoint selection """
# use geometric interpretation
def dp_waypoint_selection(
    env=None,
    actions=None,
    gt_states=None,
    err_threshold=None,
    initial_states=None,
    remove_obj=None,
    pos_only=False,
):
    if actions is None:
        actions = copy.deepcopy(gt_states)
    elif gt_states is None:
        gt_states = copy.deepcopy(actions)
        
    num_frames = len(actions)

    # make the last frame a waypoint
    initial_waypoints = [num_frames - 1]

    # make the frames of gripper open/close waypoints
    """if not pos_only:
        for i in range(num_frames - 1):
            if actions[i, -1] != actions[i + 1, -1]:
                initial_waypoints.append(i)
                # initial_waypoints.append(i + 1)
        initial_waypoints.sort()"""

    # Memoization table to store the waypoint sets for subproblems
    memo = {}

    func = (
        fast_geometric_waypoint_trajectory
    )    
    distance_func = (
        get_all_pos_only_geometric_distance_gpu if pos_only
        else get_all_geometric_distance_gpu )
    all_distance = distance_func(gt_states)
    
    # Check if err_threshold is too small, then return all points as waypoints
    min_error = func(actions, gt_states, list(range(1, num_frames)), all_distance)
    if err_threshold < min_error:
        print("Error threshold is too small, returning all points as waypoints.")
        return list(range(1, num_frames))
    
    # Initialize the memoization table
    for i in range(num_frames):
        memo[i] = (0, [])

    memo[1] = (1, [1])

    # Populate the memoization table using an iterative bottom-up approach
    for i in range(2, num_frames):
        min_waypoints_required = float("inf")
        best_waypoints = []

        for k in range(1, i):
            # waypoints are relative to the subsequence
            waypoints = [j - k for j in initial_waypoints if j >= k and j < i] + [i - k]

            total_traj_err = func(
                actions=actions[k : i + 1],
                gt_states=gt_states[k : i + 1],
                waypoints=waypoints,
                all_distance = all_distance[k : i + 1,k : i + 1,k : i + 1]
            )

            if total_traj_err < err_threshold:
                subproblem_waypoints_count, subproblem_waypoints = memo[k]
                total_waypoints_count = 1 + subproblem_waypoints_count
                if total_waypoints_count < min_waypoints_required:
                    min_waypoints_required = total_waypoints_count
                    best_waypoints = subproblem_waypoints + [i]
        memo[i] = (min_waypoints_required, best_waypoints)
        
    min_waypoints_count, waypoints = memo[num_frames - 1]
    waypoints += initial_waypoints
    # remove duplicates
    waypoints = list(set(waypoints))
    waypoints.sort()
    total_traj_err = func(
                actions=actions,
                gt_states=gt_states,
                waypoints=waypoints,
                all_distance = all_distance
            )
    """print(
        f"Minimum number of waypoints: {len(waypoints)} \tTrajectory Error: {total_traj_err}"
    )
    print(f"waypoint positions: {waypoints}")"""

    return waypoints

def calculate_weights_from_entropy(actions_entropy):
    # TODO: adjust normalization of the entropy
    # actions_entropy = (actions_entropy-np.min(actions_entropy))/(np.max(actions_entropy)-np.min(actions_entropy))
    entropy_weights = np.exp(actions_entropy*20) # |2 0.001 55% 1.93|1 0.0012 60% 2.02 no gripper|
    # print('entropy',np.max(actions_entropy),np.min(actions_entropy),'weights',np.max(entropy_weights),np.min(entropy_weights))
    return entropy_weights 

# use geometric interpretation
def dp_entropy_waypoint_selection(
    env=None,
    actions=None,
    gt_states=None,
    err_threshold=None,
    initial_states=None,
    remove_obj=None,
    pos_only=False,
):  
    entropy = actions[:,-1:]
    actions = actions[:,:-1]
    if gt_states is None:
        gt_states = copy.deepcopy(actions)
        
    num_frames = len(actions)
    entropy_weights = calculate_weights_from_entropy(entropy)
    all_err_threshold = err_threshold*entropy_weights
    # make the last frame a waypoint
    initial_waypoints = [num_frames - 1]

    # make the frames of gripper open/close waypoints
    """if not pos_only:
        for i in range(num_frames - 1):
            if actions[i, -1] != actions[i + 1, -1]:
                initial_waypoints.append(i)
                # initial_waypoints.append(i + 1)
        initial_waypoints.sort()"""

    # Memoization table to store the waypoint sets for subproblems
    memo = {}

    func = (
        fast_geometric_waypoint_trajectory
    )    
    distance_func = (
        get_all_pos_only_geometric_distance_gpu if pos_only
        else get_all_geometric_distance_gpu )
    all_distance = distance_func(gt_states)
    
    # Check if err_threshold is too small, then return all points as waypoints
    min_error = func(actions, gt_states, list(range(1, num_frames)), all_distance)
    if err_threshold < min_error:
        print("Error threshold is too small, returning all points as waypoints.")
        return list(range(1, num_frames))
    
    # Initialize the memoization table
    for i in range(num_frames):
        memo[i] = (0, [])

    memo[1] = (1, [1])

    # Populate the memoization table using an iterative bottom-up approach
    for i in range(2, num_frames):
        min_waypoints_required = float("inf")
        best_waypoints = []

        for k in range(1, i):
            # waypoints are relative to the subsequence
            waypoints = [j - k for j in initial_waypoints if j >= k and j < i] + [i - k]

            total_traj_err, all_traj_err = func(
                actions=actions[k : i + 1],
                gt_states=gt_states[k : i + 1],
                waypoints=waypoints,
                all_distance = all_distance[k : i + 1,k : i + 1,k : i + 1],
                return_list = True
            )

            if (np.array(all_traj_err)<=all_err_threshold[k:i+1]).all():
                subproblem_waypoints_count, subproblem_waypoints = memo[k]
                total_waypoints_count = 1 + subproblem_waypoints_count
                if total_waypoints_count < min_waypoints_required:
                    min_waypoints_required = total_waypoints_count
                    best_waypoints = subproblem_waypoints + [i]
        memo[i] = (min_waypoints_required, best_waypoints)
        
    min_waypoints_count, waypoints = memo[num_frames - 1]
    waypoints += initial_waypoints
    # remove duplicates
    waypoints = list(set(waypoints))
    waypoints.sort()
    total_traj_err = func(
                actions=actions,
                gt_states=gt_states,
                waypoints=waypoints,
                all_distance = all_distance
            )
    """print(
        f"Minimum number of waypoints: {len(waypoints)} \tTrajectory Error: {total_traj_err}"
    )
    print(f"waypoint positions: {waypoints}")"""

    return waypoints

def total_state_err(err_dict):
    return err_dict["err_pos"] + err_dict["err_quat"]


def total_traj_err(err_list):
    # return np.mean(err_list)
    return np.max(err_list)


# Utilize GPU to accelerate calculation
def get_all_pos_only_geometric_distance_gpu(gt_states):
    """Compute the geometric trajectory from the waypoints using PyTorch"""

    gt_states = torch.tensor(gt_states, dtype=torch.float32, device=device)
    n = gt_states.size(0)
    d = gt_states.size(1)
    
    # Expand dimensions for broadcasting
    gt_states_i = gt_states.unsqueeze(1).unsqueeze(2)  # Shape: (n, 1, 1, d)
    gt_states_j = gt_states.unsqueeze(0).unsqueeze(2)  # Shape: (1, n, 1, d)
    gt_states_k = gt_states.unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, n, d)
    
    # Calculate line vectors
    line_vector = gt_states_k - gt_states_j  # Shape: (1, n, n, d)
    point_vector = gt_states_i - gt_states_j  # Shape: (n, n, 1, d)
    
    # Calculate t values
    dot_product = torch.sum(point_vector * line_vector, dim=-1)
    norm_line = torch.sum(line_vector * line_vector, dim=-1)+1e-8
    t = dot_product / norm_line
    t = torch.clamp(t, 0, 1)
    
    # Calculate projections
    projection = gt_states_j + t.unsqueeze(-1) * line_vector
    
    # Calculate distances
    distances = torch.norm(gt_states_i - projection, dim=-1)
    return distances.cpu().numpy()

def get_all_geometric_distance_gpu(gt_states):
    """Compute the geometric trajectory from the waypoints using PyTorch"""
    gt_pos = gt_states[:,:3]
    gt_quat = gt_states[:,3:6]
    gt_pos = torch.tensor(gt_pos, dtype=torch.float32, device=device)
    gt_quat = torch.tensor(gt_quat, dtype=torch.float32, device=device)
    n = gt_pos.size(0)
    pos_d = gt_pos.size(1)
    quat_d = gt_quat.size(1)
    
    # Expand dimensions for broadcasting
    gt_pos_i = gt_pos.unsqueeze(1).unsqueeze(2)  # Shape: (n, 1, 1, d)
    gt_pos_j = gt_pos.unsqueeze(0).unsqueeze(2)  # Shape: (1, n, 1, d)
    gt_pos_k = gt_pos.unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, n, d)
    
    # Calculate line vectors
    line_vector = gt_pos_k - gt_pos_j  # Shape: (1, n, n, d)
    point_vector = gt_pos_i - gt_pos_j  # Shape: (n, n, 1, d)
    
    # Calculate t values
    dot_product = torch.sum(point_vector * line_vector, dim=-1)
    norm_line = torch.sum(line_vector * line_vector, dim=-1)+1e-8
    t = dot_product / norm_line
    t = torch.clamp(t, 0, 1)
    
    # Calculate projections
    projection = gt_pos_j + t.unsqueeze(-1) * line_vector
    
    # Calculate distances
    pos_distances = torch.norm(gt_pos_i - projection, dim=-1)
    # Perform distance on gt_quat
    # Expand dimensions for broadcasting
    gt_quat_i = gt_quat.unsqueeze(1).unsqueeze(2)  # Shape: (n, 1, 1, d)
    gt_quat_j = gt_quat.unsqueeze(0).unsqueeze(2)  # Shape: (1, n, 1, d)
    gt_quat_k = gt_quat.unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, n, d)
    
    # Calculate line vectors
    line_vector = gt_quat_k - gt_quat_j  # Shape: (1, n, n, d)
    point_vector = gt_quat_i - gt_quat_j  # Shape: (n, n, 1, d)
    
    # Calculate t values
    dot_product = torch.sum(point_vector * line_vector, dim=-1)
    norm_line = torch.sum(line_vector * line_vector, dim=-1)+1e-8
    t = dot_product / norm_line
    t = torch.clamp(t, 0, 1)
    
    # Calculate projections
    projection = gt_quat_j + t.unsqueeze(-1) * line_vector
    
    # Calculate distances
    quat_distances = torch.norm(gt_quat_i - projection, dim=-1)
    distances = pos_distances + quat_distances
    return distances.cpu().numpy()

def fast_geometric_waypoint_trajectory(actions,gt_states,waypoints,all_distance,return_list=False):
    if waypoints[0] != 0:
        waypoints = [0] + waypoints
    
    state_err = np.empty(0, dtype=np.float32)
    n_segments = len(waypoints) - 1
    if n_segments>0:
        for i in range(n_segments):
            state_err=np.concatenate((state_err,all_distance[waypoints[i]:waypoints[i + 1],waypoints[i],waypoints[i+1]]))

    state_err=np.concatenate((state_err,np.zeros(1,dtype=np.float32)))  
    if return_list:
        return total_traj_err(state_err), state_err
    else:
        return total_traj_err(state_err)