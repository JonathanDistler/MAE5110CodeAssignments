import numpy as np

#from the wikipedia, h i sthe step, then there are four stages (four orders, maybe)
def rk4(dynamics, is_touching, reset_params, initial_state, timestep, sim_time, params, num_contacts=None): 
    n_timesteps = int(sim_time / timestep) + 1

    time_traj = np.arange(n_timesteps) * timestep

    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    contact_counter=0
    theta_dot_pos=[]
    contact_indices=[]
    pre_impact_states=[]
    post_impact_states=[]


    for step, t in enumerate(time_traj[:-1]):
        current_state=state_traj[:,step]


        #implemented from the wikipedia article using t as the tn as per the ed discussion and timestep as h
        k1=dynamics(t,state_traj[:, step], params)
        k2=dynamics(t+timestep/2,state_traj[:,step]+k1*timestep/2,params)
        k3=dynamics(t+timestep/2,state_traj[:,step]+k2*timestep/2,params)
        k4=dynamics(t+timestep,state_traj[:,step]+timestep*k3,params)
        new_state=(current_state + timestep / 6 * (k1 + 2*k2 + 2*k3 + k4))
        #state_traj[:, step + 1] = (state_traj[:, step] + timestep / 6 * (k1 + 2*k2 + 2*k3 + k4)) - wrapped up in new_state
        if (is_touching(new_state,params)):
            pre_impact_state=new_state.copy()
            new_state=reset_params(new_state,params)
            post_impact_state=new_state.copy()
            contact_counter+=1
            theta_dot_pos.append(new_state[1])
            contact_indices.append(step + 1)
            pre_impact_states.append(pre_impact_state)
            post_impact_states.append(post_impact_state)
        state_traj[:,step+1]=new_state

        if num_contacts is not None:
            if contact_counter>=num_contacts:
                return time_traj, state_traj, theta_dot_pos, contact_indices,pre_impact_states, post_impact_states


    return time_traj, state_traj, theta_dot_pos, contact_indices,pre_impact_states, post_impact_states

def euler(dynamics, initial_state, timestep, sim_time, params):
    n_timesteps = int(sim_time / timestep) + 1

    time_traj = np.arange(n_timesteps) * timestep

    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    for step, t in enumerate(time_traj[:-1]):
        state_traj[:, step + 1] = state_traj[:, step] + timestep * dynamics(t, state_traj[:, step], params)

    return time_traj, state_traj