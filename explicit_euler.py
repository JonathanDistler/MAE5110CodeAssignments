import numpy as np

#from the wikipedia, h i sthe step, then there are four stages (four orders, maybe)
def integrate(dynamics, initial_state, timestep, sim_time, params):
    n_timesteps = int(sim_time / timestep) + 1

    time_traj = np.arange(n_timesteps) * timestep

    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    for step, t in enumerate(time_traj[:-1]):
        #implemented from the wikipedia article using t as the tn as per the ed discussion and timestep as h
        k1=dynamics(t,state_traj[:, step], params)
        k2=dynamics(t+timestep/2,state_traj[:,step]+k1*timestep/2,params)
        k3=dynamics(t+timestep/2,state_traj[:,step]+k2*timestep/2,params)
        k4=dynamics(t+timestep,state_traj[:,step]+timestep*k3,params)
        state_traj[:, step + 1] = (state_traj[:, step] + timestep / 6 * (k1 + 2*k2 + 2*k3 + k4))

    return time_traj, state_traj