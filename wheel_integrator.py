import numpy as np 
 
#from the wikipedia, h is the step, then there are four stages (four orders, maybe) 

# RK4 INTEGRATOR 
def rk4(dynamics, is_touching, reset_params, initial_state, timestep, time_step_upper_bound, sim_time, params, num_contacts=None, theta_contact=None, theta_tolerance=0.05): 
 
    # INITIALIZE TRAJECTORIES 
    # These are lists because the timestep can change during the simulation 
    time_traj = [0.0] 
    state_traj = [np.array(initial_state, dtype=float)] 
 
    contact_counter = 0 
    theta_dot_pos = [] 
    contact_indices = [] 
    pre_impact_states = [] 
    post_impact_states = [] 
 
    t = 0.0 
 
    while t < sim_time: 
 
        current_state = state_traj[-1]
        
        # DETERMINE THE APPROPRIATE TIMESTEP 
        if theta_contact is not None: 
 
            # Distance from the current leg angle to the contact angle 
            theta_distance = abs(theta_contact - current_state[0]) 
 
            # Use the smaller timestep when approaching contact 
            if theta_distance <= theta_tolerance and current_state[1] > 0: 
                current_timestep = timestep 
 
            # Otherwise use the larger timestep to speed up simulation 
            else: 
                current_timestep = time_step_upper_bound 
 
        else: 
            # If no contact angle is provided, always use the normal timestep 
            current_timestep = timestep 
 
        # DO NOT STEP PAST THE END OF THE SIMULATION 
        current_timestep = min(current_timestep, sim_time - t) 
 
        # RK4 CALCULATION 
        k1 = dynamics(t, current_state, params) 
        k2 = dynamics(t + current_timestep / 2, current_state + k1 * current_timestep / 2, params) 
        k3 = dynamics(t + current_timestep / 2, current_state + k2 * current_timestep / 2, params) 
        k4 = dynamics(t + current_timestep, current_state + k3 * current_timestep, params) 
        new_state = (current_state + current_timestep / 6 * (k1 + 2*k2 + 2*k3 + k4)) 
 
        # IMPACT DETECTION 
        if is_touching(new_state, params): 
            pre_impact_state = new_state.copy() 
            new_state = reset_params(new_state, params) 
 
            post_impact_state = new_state.copy() 
            contact_counter += 1 
            theta_dot_pos.append(new_state[1]) 
 
            # Index of the new state in the trajectory 
            contact_indices.append(len(state_traj)) 
            pre_impact_states.append(pre_impact_state) 
            post_impact_states.append(post_impact_state) 
 
        # SAVE NEW STATE 
        t += current_timestep 
        time_traj.append(t) 
        state_traj.append(new_state) 
 
        # CHECK NUMBER OF CONTACTS 
        if num_contacts is not None: 
 
            if contact_counter >= num_contacts: 
                return (np.array(time_traj), np.array(state_traj), theta_dot_pos, contact_indices, pre_impact_states, post_impact_states) 
 
    # RETURN RESULTS 
    return (np.array(time_traj), np.array(state_traj), theta_dot_pos, contact_indices, pre_impact_states, post_impact_states) 


# EULER INTEGRATOR 
def euler(dynamics, initial_state, timestep, sim_time, params): 
 
    n_timesteps = int(sim_time / timestep) + 1 
 
    time_traj = np.arange(n_timesteps) * timestep 
 
    state_traj = np.zeros((2, n_timesteps)) 
 
    state_traj[:, 0] = initial_state 
 
    for step, t in enumerate(time_traj[:-1]): 
 
        state_traj[:, step + 1] = ( 
            state_traj[:, step] 
            + timestep * dynamics( 
                t, 
                state_traj[:, step], 
                params 
            ) 
        ) 
 
    return time_traj, state_traj