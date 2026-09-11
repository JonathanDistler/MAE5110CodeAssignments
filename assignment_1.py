"""
Currently, there are no graphs to represent any of the deliverables for the assignment. And, the markdown file associated
with the assignment is blank, which doesn't provide a ton of direction. The goal for my review was to comment on syntax and point out 
possible directions that could be used to push in the right direction. I think it would be helpful to include the following:

* fixed point function
* walking threshold function
* limit cycle function
* return map function
* roa function
* inclination sweep function
* spoke sweep function
"""

from concurrent.futures import ProcessPoolExecutor
#unusual library, could be helpful to add a brief overview of what it does 

import numpy as np
from integrators import ivp_with_guard as integrator
from models import rimless_wheel as model
from plot import animate_rimless_wheel


#none of the parameters are defined explicitly in this file 
# e.g. params={"gravity": 9.81, "slope_angle": 0.1, "half_spoke_angle": 0.2, "spoke_length": 1.0}

params = model.generate_params()
initial_state = np.array([0.0, 1.0])
#seems like a pretty small sime time, I had done 20 s, but there's probably a sweet spot in the middle
#I also found it helpful to keep track of the number of contacts for RoA
sim_time = 2
#also, there isn'a timestep defined anywhere - it could be helpful to define two separate
#timesteps: one for trajectories far away from the poincare plane and one within a threshold of the poincare plane (e.g. 0.01 and 0.001) - this could save some computation time and improve accuracy

#never actually call the "run_sim" function
def run_sim(integrator, timestep):
    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state
    for step, t in enumerate(time_traj[:-1]):
        state_traj[:, step+1] = integrator(t, state_traj[:, step], params, timestep, model)

    return state_traj

# From the underactuated page on MIT, there are some good resources on fixed points and walking thresholds
# which could save some computation time and automatically negate some of the initial conditions that are not feasible. I had done a similar thing in my own code, but it was a bit more manual and less automated.

# I found it helpful to have a calculate_limit_cycle, return_map, calculate_floquet_multiplier, plot_roa, and sweep functions
# Floquet is typically the poincare function of a Jacobian, but it can be approximate with a simple slope formula


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
#41x41 space
n_theta=41
n_theta_dot=41
sim_time=15.0,
#what do the workers do? maybe a vrief description
n_workers=8
gamma = params["slope_angle"]
alpha = params["half_spoke_angle"]
theta_range = (gamma - alpha - 0.3, gamma + alpha + 0.3)
theta_dot_range = (-4.0, 4.0)
#had been unfamiliar with *, but very clever! 
theta_vals = np.linspace(*theta_range, n_theta)
theta_dot_vals = np.linspace(*theta_dot_range, n_theta_dot)

#now, just need to iterate over all of them and call the run_sim function, along with any set of relevant plot functions
grid_points = [
    (theta0, theta_dot0, params, sim_time)
    for theta_dot0 in theta_dot_vals
    for theta0 in theta_vals
]
#nothing plotted, definitely a reasonable sweep range (how did you decide on the range of theta-dot and theta?); it could be helpful to allude to
#the process in the markdown 


# Sweep and Simulate

