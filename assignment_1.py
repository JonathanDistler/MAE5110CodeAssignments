from concurrent.futures import ProcessPoolExecutor

import numpy as np
from integrators import ivp_with_guard as integrator
from models import rimless_wheel as model
from plot import animate_rimless_wheel



params = model.generate_params()
initial_state = np.array([0.0, 1.0])
sim_time = 2

def run_sim(integrator, timestep):
    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state
    for step, t in enumerate(time_traj[:-1]):
        state_traj[:, step+1] = integrator(t, state_traj[:, step], params, timestep, model)

    return state_traj


## Sanity Check
#time_traj, state_traj, collision_times, collision_states = integrator(
#    initial_state=[0.0, 1.0],
#    params=params,
#    model=model,
#    sim_time=10.0,
#)
#anim = animate_rimless_wheel(time_traj, state_traj, params) # animator made with ai


#final_state = state_traj[:, -1]



# RoA Sweep
# Get RoA Grid Points
n_theta=41
n_theta_dot=41
sim_time=15.0,
n_workers=8
gamma = params["slope_angle"]
alpha = params["half_spoke_angle"]
theta_range = (gamma - alpha - 0.3, gamma + alpha + 0.3)
theta_dot_range = (-4.0, 4.0)

theta_vals = np.linspace(*theta_range, n_theta)
theta_dot_vals = np.linspace(*theta_dot_range, n_theta_dot)

grid_points = [
    (theta0, theta_dot0, params, sim_time)
    for theta_dot0 in theta_dot_vals
    for theta0 in theta_vals
]



# Sweep and Simulate

