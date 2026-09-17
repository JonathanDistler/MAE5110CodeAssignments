from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from models import inverted_pendulum_walker as model


# Fixed controls for this visualization example.
params = {
    "gravity": 9.81,  # m/s^2
    "length": 1.0,  # m
    "mass": 1.0,  # kg
    "incline": 0.06,  # rad
    "angle_of_attack": np.pi / 8,  # rad
    "ankle_torque": 0.0,  # N m
    "damping_coeff": 0.0 # damping coefficient (kg*m^2/s)
}

initial_state = np.array([0.0, 3.0])
timestep = 1e-4
sim_time = 3.0

# CONTROL BOUNDS
def get_control_bounds(params):

    length=params["length"]
    mass=params["mass"]
    gravity=params["gravity"]

    angle_of_attack_min=np.pi/8
    angle_of_attack_max=np.pi/7

    tau_min=-.1*length*mass*gravity
    tau_max=.05*length*mass*gravity

    return angle_of_attack_min, angle_of_attack_max, tau_min, tau_max


# BALANCING CONTROLLER
def feedback_linearization_controller(state, params, kp=2.0, kd=0.5):
    #feedback-linearization controller with desired closed-loop dynamics
    length=params["length"]
    mass=params["mass"]
    gravity=params["gravity"]
    damping=params["damping_coeff"]

    theta=state[0]
    theta_dot=state[1]

    desired_acceleration=-kp*theta-kd*theta_dot
    required_torque=(mass*length**2*desired_acceleration - mass*gravity*length*np.sin(theta) + damping*theta_dot)

    return required_torque


# RoA
def in_RoA_helper(state, params, kp=2.0, kd=0.5):

    required_torque=feedback_linearization_controller(state, params, kp=kp, kd=kd)
    _, _, tau_min, tau_max=get_control_bounds(params)

    #state is inside the balancing RoA if the required controller torque is achievable
    return tau_min <= required_torque <= tau_max


# POINCARE SECTION
def poincare_section_event(prev_state, next_state, params):

    theta_prev=prev_state[0]
    theta_next=next_state[0]

    #defines the poincare section as passing through the vertical upright position
    #walker leaves at theta<0 and moves towards theta>0
    return (theta_prev < 0 and theta_next >= 0)


#used to interpolate between grid points to find exact position of an event
def interpolate_state(previous_state, next_state, target_theta):

    theta_prev=previous_state[0]
    theta_next=next_state[0]

    if theta_next == theta_prev:
        return next_state.copy()

    fraction=((target_theta-theta_prev)/(theta_next-theta_prev))
    return previous_state + fraction*(next_state-previous_state)


# POINCARE MAP
def poincare_map(theta_dot, alpha, params, timestep=1e-4):

    step_params=params.copy()
    step_params["angle_of_attack"]=alpha

    #ankle controller is off during walking
    step_params["ankle_torque"]=0.0
    gamma=step_params["incline"]

    #touchdown angle for the inclined surface
    theta_td=gamma+alpha

    #start on the Poincare section theta=0
    state=np.array([0.0, theta_dot])
    max_steps=20000

    # FIND TOUCHDOWN
    touchdown_found=False

    for i in range(max_steps):
        next_state=(state + timestep*model.dynamics(0, state,step_params))

        if (state[0] < theta_td and next_state[0] >= theta_td):
            touchdown_state=interpolate_state(state, next_state, theta_td)
            state=model.event_dynamics(touchdown_state, step_params)
            touchdown_found=True
            break

        state=next_state

    if not touchdown_found:
        return np.nan

    # FIND NEXT POINCARE SECTION
    for j in range(max_steps):
        next_state=(state + timestep*model.dynamics(0, state, step_params))

        if poincare_section_event(state, next_state, step_params):
            section_state=interpolate_state(state, next_state, 0.0)

            #only accept the forward-moving branch
            if section_state[1] > 0:
                return section_state[1]

            return np.nan

        state=next_state

    return np.nan


# ANGLE OF ATTACK POLICY
def angle_of_attack_policy(state, params):

    theta=state[0]
    theta_dot=state[1]

    controller_alpha=np.pi/8
    aot_min, aot_max, __, __=get_control_bounds(params)

    alpha=np.clip(controller_alpha, aot_min, aot_max)

    return alpha


# CALCULATE BALANCING RoA
print("BEGINNING BALANCING RoA SEARCH")

#small grid around the useful state range
theta_grid=np.linspace(-np.pi/2, np.pi/2, 81)
theta_dot_grid_roa=np.linspace(-6, 6, 121)

#only considering two options for RoA - in RoA or not in RoA
roa_grid=np.zeros((len(theta_dot_grid_roa), len(theta_grid)), dtype=bool)

#controller gains
kp=2.0
kd=0.5

for i, theta_dot in enumerate(theta_dot_grid_roa):
    for j, theta in enumerate(theta_grid):
        state=np.array([theta, theta_dot])
        roa_grid[i,j]=in_RoA_helper(state, params, kp=kp, kd=kd)

number_roa_states=np.sum(roa_grid)
number_total_states=roa_grid.size

print("BALANCING RoA SEARCH COMPLETE")
print(f"RoA states: {number_roa_states}/{number_total_states}")

# OUTPUT DIRECTORY
output_dir=Path("output/assignment_2")
output_dir.mkdir(parents=True, exist_ok=True)

# PLOT BALANCING RoA
fig, ax=plt.subplots(figsize=(8,6), layout="constrained")
ax.contourf(theta_grid, theta_dot_grid_roa, roa_grid.astype(float), levels=[0.5,1.5])
ax.axvline(0, linestyle="--")
ax.axhline(0, linestyle="--")
ax.set_xlabel(r"$\theta$ (rad)")
ax.set_ylabel(r"$\dot{\theta}$ (rad/s)")
ax.set_title("Balancing Controller Region of Attraction")
fig.savefig(output_dir / "01_balancing_roa.png", dpi=300)
plt.close(fig)


# RoA LOOKUP FUNCTION
def state_is_in_roa(state, params):
    #use the actual continuous state instead of the nearest RoA grid point
    return in_RoA_helper(state, params, kp=kp, kd=kd)


def poincare_state_is_in_roa(theta_dot):
    state=np.array([0.0, theta_dot])
    return state_is_in_roa(state, params)


# BUILD POINCARE TABLE
def build_poincare_table(theta_dot_grid, alpha_grid, params, timestep=1e-4, print_progress=True):

    poincare_table=np.full((len(theta_dot_grid), len(alpha_grid)), np.nan)

    for i, theta_dot in enumerate(theta_dot_grid):
        if print_progress:
            print(f"Searching theta_dot = {theta_dot:.4f} rad/s, {i+1}/{len(theta_dot_grid)}")

        for j, alpha in enumerate(alpha_grid):
            poincare_table[i,j]=poincare_map(theta_dot, alpha, params, timestep=timestep)

    return poincare_table


# BUILD MINIMUM-STEP BACKWARD REACHABILITY
def build_minimum_steps(theta_dot_grid, alpha_grid, poincare_table):

    #steps_to_roa[i] stores the minimum number of walking steps needed to reach RoA 0 means already in RoA and -1 means not reachable
    steps_to_roa=np.full(len(theta_dot_grid),-1,dtype=int)
    policy_alpha=np.full(len(theta_dot_grid),np.nan)

    for i, theta_dot in enumerate(theta_dot_grid):
        if poincare_state_is_in_roa(theta_dot):
            steps_to_roa[i]=0
            policy_alpha[i]=alpha_grid[0]

    max_backward_iterations=len(theta_dot_grid)

    for iteration in range(max_backward_iterations):
        previous_steps_to_roa=steps_to_roa.copy()

        for i, theta_dot in enumerate(theta_dot_grid):
            if previous_steps_to_roa[i] >= 0:
                continue

            for j, alpha in enumerate(alpha_grid):
                next_theta_dot=poincare_table[i,j]

                if not np.isfinite(next_theta_dot):
                    continue

                if next_theta_dot <= 0:
                    continue

                next_index=np.argmin(np.abs(theta_dot_grid-next_theta_dot))

                if previous_steps_to_roa[next_index] >= 0:
                    steps_to_roa[i]=previous_steps_to_roa[next_index]+1
                    policy_alpha[i]=alpha
                    break

        new_states=np.sum((steps_to_roa >= 0) & (previous_steps_to_roa < 0))
        print(f"Backward iteration {iteration+1}: {np.sum(steps_to_roa >= 0)} reachable states (+{new_states})")

        if np.array_equal(steps_to_roa,previous_steps_to_roa):
            break

    return steps_to_roa, policy_alpha


# BUILD MAXIMUM-STEP BACKWARD REACHABILITY
def build_maximum_steps(theta_dot_grid, alpha_grid, poincare_table, max_horizon=20):

    #maximum_steps_to_roa stores the longest viable route to RoA found within max_horizon 0 means already in RoA and -1 means no route to RoA was found
    maximum_steps_to_roa=np.full(len(theta_dot_grid),-1,dtype=int)
    maximum_policy_alpha=np.full((len(theta_dot_grid),max_horizon),np.nan)

    for i, theta_dot in enumerate(theta_dot_grid):
        if poincare_state_is_in_roa(theta_dot):
            maximum_steps_to_roa[i]=0

    #The horizon is intentionally finite so that cycles do not create an artificial infinity. The result therefore means the longest route found within max_horizon walking steps.
    for horizon in range(1,max_horizon+1):
        previous_maximum_steps=maximum_steps_to_roa.copy()
        new_maximum_steps=maximum_steps_to_roa.copy()
        new_policy=maximum_policy_alpha.copy()

        for i, theta_dot in enumerate(theta_dot_grid):
            if previous_maximum_steps[i] == 0:
                continue

            for j, alpha in enumerate(alpha_grid):
                next_theta_dot=poincare_table[i,j]

                if not np.isfinite(next_theta_dot):
                    continue

                if next_theta_dot <= 0:
                    continue

                next_index=np.argmin(np.abs(theta_dot_grid-next_theta_dot))

                if previous_maximum_steps[next_index] >= 0:
                    candidate=previous_maximum_steps[next_index]+1

                    if candidate > new_maximum_steps[i]:
                        new_maximum_steps[i]=candidate
                        new_policy[i,horizon-1]=alpha

        maximum_steps_to_roa=new_maximum_steps
        maximum_policy_alpha=new_policy

    return maximum_steps_to_roa, maximum_policy_alpha


# LOOKUP POLICY
def lookup_alpha(theta_dot, theta_dot_grid, steps_to_roa, policy_alpha):
    index=np.argmin(np.abs(theta_dot_grid-theta_dot))

    if steps_to_roa[index] < 0:
        return np.nan

    return policy_alpha[index]


# INTERPOLATE STEPS TO RoA
#interpolates between grid lines for higher fidelity 
def interpolated_steps_to_roa(theta_dot, theta_dot_grid, steps_to_roa):

    if poincare_state_is_in_roa(theta_dot):
        return 0

    if (theta_dot < theta_dot_grid[0] or theta_dot > theta_dot_grid[-1]):
        return -1

    upper_index=np.searchsorted(theta_dot_grid,theta_dot)

    if upper_index == 0:
        lower_index=0
        upper_index=0
    elif upper_index >= len(theta_dot_grid):
        lower_index=len(theta_dot_grid)-1
        upper_index=len(theta_dot_grid)-1
    else:
        lower_index=upper_index-1

    lower_steps=steps_to_roa[lower_index]
    upper_steps=steps_to_roa[upper_index]

    if (lower_steps < 0 or upper_steps < 0):
        return -1

    if lower_index == upper_index:
        return int(lower_steps)

    lower_velocity=theta_dot_grid[lower_index]
    upper_velocity=theta_dot_grid[upper_index]
    fraction=(theta_dot-lower_velocity)/(upper_velocity-lower_velocity)
    interpolated_steps=lower_steps+fraction*(upper_steps-lower_steps)

    return int(interpolated_steps+0.5)


# WALK ONE STEP
def integrate_to_touchdown(state, alpha, params, timestep=1e-4):

    step_params=params.copy()
    step_params["angle_of_attack"]=alpha
    step_params["ankle_torque"]=0.0
    gamma=step_params["incline"]
    theta_td=gamma+alpha
    max_steps=20000

    for i in range(max_steps):
        next_state=(state+timestep*model.dynamics(0,state,step_params))

        if (state[0] < theta_td and next_state[0] >= theta_td):
            touchdown_state=interpolate_state(state,next_state,theta_td)
            return model.event_dynamics(touchdown_state,step_params)

        state=next_state

    return None


def integrate_to_poincare(state, params, timestep=1e-4):

    step_params=params.copy()
    step_params["ankle_torque"]=0.0
    max_steps=20000

    for i in range(max_steps):
        next_state=(state+timestep*model.dynamics(0,state,step_params))

        if poincare_section_event(state,next_state,step_params):
            section_state=interpolate_state(state,next_state,0.0)

            if section_state[1] > 0:
                return section_state

            return None

        state=next_state

    return None


# SIMULATE A SPECIFIC POLICY
def simulate_policy(initial_state, params, theta_dot_grid, alpha_grid, policy_alpha, steps_to_roa, maximum_steps_to_roa, max_walking_steps=20, timestep=1e-4, use_max_policy=False):

    state=initial_state.copy()
    state_history=[state.copy()]
    alpha_history=[]
    poincare_history=[state[1]]
    completed_steps=0

    for step in range(max_walking_steps):
        if state_is_in_roa(state,params):
            print(f"Entered RoA after {completed_steps} walking steps")
            return (state_history,alpha_history,poincare_history,completed_steps,True)

        params["ankle_torque"]=0.0
        theta_dot=state[1]
        index=np.argmin(np.abs(theta_dot_grid-theta_dot))

        if use_max_policy:
            #select an alpha that keeps the walker on a longest viable route
            alpha=np.nan
            desired_remaining=maximum_steps_to_roa[index]

            for j, alpha_candidate in enumerate(alpha_grid):
                next_theta_dot=poincare_table[index,j]

                if not np.isfinite(next_theta_dot):
                    continue

                if next_theta_dot <= 0:
                    continue

                next_index=np.argmin(np.abs(theta_dot_grid-next_theta_dot))

                if maximum_steps_to_roa[next_index] == desired_remaining-1:
                    alpha=alpha_candidate
                    break

            #fallback if the exact longest route is not available from the rounded state
            if not np.isfinite(alpha):
                for j, alpha_candidate in enumerate(alpha_grid):
                    next_theta_dot=poincare_table[index,j]

                    if not np.isfinite(next_theta_dot):
                        continue

                    if next_theta_dot <= 0:
                        continue

                    next_index=np.argmin(np.abs(theta_dot_grid-next_theta_dot))

                    if maximum_steps_to_roa[next_index] >= 0:
                        alpha=alpha_candidate
                        break
        else:
            alpha=lookup_alpha(theta_dot,theta_dot_grid,steps_to_roa,policy_alpha)

        if not np.isfinite(alpha):
            print("Current state is not reachable by the lookup table")
            return (state_history,alpha_history,poincare_history,completed_steps,False)

        print(f"step {step+1}: alpha = {alpha:.5f}, theta_dot = {theta_dot:.5f}")
        alpha_history.append(alpha)

        touchdown_state=integrate_to_touchdown(state,alpha,params,timestep=timestep)

        if touchdown_state is None:
            print("Touchdown was not detected")
            return (state_history,alpha_history,poincare_history,completed_steps,False)

        next_state=integrate_to_poincare(touchdown_state,params,timestep=timestep)

        if next_state is None:
            print("Next Poincare section was not detected")
            return (state_history,alpha_history,poincare_history,completed_steps,False)

        state=next_state.copy()
        state_history.append(state.copy())
        poincare_history.append(state[1])
        completed_steps+=1

        print(f"Next Poincare theta_dot = {state[1]:.5f}")

    print("Maximum walking steps reached without entering the RoA")
    return (state_history,alpha_history,poincare_history,completed_steps,False)


# SELECT AN INITIAL CONDITION WITH A LONG VIABLE ROUTE
def select_three_step_initial_condition(theta_dot_grid, minimum_steps, maximum_steps):

    #First look for a state that genuinely requires at least 3 steps under the minimum-step policy.
    candidates=np.where(minimum_steps >= 3)[0]

    if len(candidates) > 0:
        index=candidates[np.argmax(minimum_steps[candidates])]
        return index,"minimum"

    #If no state requires 3 steps, find a state that has a viable route of at least 3 steps.
    candidates=np.where(maximum_steps >= 3)[0]

    if len(candidates) > 0:
        index=candidates[np.argmax(maximum_steps[candidates])]
        return index,"maximum"

    return None,None


# PLOT WALKING TRAJECTORY
def plot_poincare_trajectory(poincare_history, output_path, title):

    fig, ax=plt.subplots(figsize=(8,6),layout="constrained")
    poincare_roa=np.array([poincare_state_is_in_roa(v) for v in poincare_history])

    if np.any(poincare_roa):
        roa_indices=np.where(poincare_roa)[0]
        ax.scatter(roa_indices,poincare_history[poincare_roa],label="Balancing RoA",zorder=5)

    ax.plot(range(len(poincare_history)),poincare_history,marker="o")

    for k, velocity in enumerate(poincare_history):
        ax.annotate(f"{velocity:.2f}",(k,velocity),xytext=(5,5),textcoords="offset points")

    ax.set_xlabel("Poincare step $k$")
    ax.set_ylabel(r"$\dot{\theta}_k$ (rad/s)")
    ax.set_title(title)
    ax.legend()
    fig.savefig(output_path,dpi=300)
    plt.close(fig)


# GENERATE WALKING GIF
def generate_walking_gif(initial_state,alpha_history,params,output_path,timestep=1e-4):

    if len(alpha_history) == 0:
        return False

    visualization_states=[]
    state=initial_state.copy()
    visualization_states.append(state.copy())
    frame_skip=50

    for step, alpha in enumerate(alpha_history):
        params["angle_of_attack"]=alpha
        params["ankle_torque"]=0.0
        gamma=params["incline"]
        theta_td=gamma+alpha
        max_steps=20000
        frame_counter=0

        for i in range(max_steps):
            next_state=(state+timestep*model.dynamics(0,state,params))

            if frame_counter % frame_skip == 0:
                visualization_states.append(next_state.copy())

            frame_counter+=1

            if (state[0] < theta_td and next_state[0] >= theta_td):
                touchdown_state=interpolate_state(state,next_state,theta_td)
                state=model.event_dynamics(touchdown_state,params)
                visualization_states.append(state.copy())
                break

            state=next_state

    if len(visualization_states) <= 1:
        return False

    fig, ax=plt.subplots(figsize=(8,6))

    def update(frame):
        ax.clear()
        model.visualize(visualization_states[frame],params,ax=ax)
        ax.set_title(f"Walking frame {frame+1}/{len(visualization_states)}")

    animation=FuncAnimation(fig,update,frames=len(visualization_states),interval=20)
    animation.save(output_path,writer=PillowWriter(fps=30))
    plt.close(fig)

    print(f"Saved {output_path} ({len(visualization_states)} frames)")
    return True


# RUN MAIN ANALYSIS
print("BEGINNING POINCARE SEARCH")

theta_dot_grid=np.linspace(0,np.sqrt(2*params["gravity"]/params["length"]),81)
alpha_grid=np.linspace(np.pi/8,np.pi/7,21)

poincare_table=build_poincare_table(theta_dot_grid,alpha_grid,params,timestep=timestep,print_progress=True)
print("POINCARE SEARCH COMPLETE")

# MINIMUM-STEP BACKWARD REACHABILITY
print("BEGINNING MINIMUM-STEP BACKWARD REACHABILITY")
steps_to_roa,policy_alpha=build_minimum_steps(theta_dot_grid,alpha_grid,poincare_table)
print("MINIMUM-STEP BACKWARD REACHABILITY COMPLETE")
print(f"Reachable states: {np.sum(steps_to_roa >= 0)}/{len(steps_to_roa)}")


# MAXIMUM-STEP BACKWARD REACHABILITY
print("BEGINNING MAXIMUM-STEP BACKWARD REACHABILITY")
max_walking_steps=20
maximum_steps_to_roa,maximum_policy_alpha=build_maximum_steps(theta_dot_grid,alpha_grid,poincare_table,max_horizon=max_walking_steps)
print("MAXIMUM-STEP BACKWARD REACHABILITY COMPLETE")
print(f"States with a viable route: {np.sum(maximum_steps_to_roa >= 0)}/{len(maximum_steps_to_roa)}")
print(f"Maximum viable walking steps found: {np.max(maximum_steps_to_roa)}")


# INTERPOLATED INITIAL STEPS
print("INTERPOLATED STEPS TO RoA")
estimated_steps=interpolated_steps_to_roa(initial_state[1],theta_dot_grid,steps_to_roa)
print(f"Initial theta_dot = {initial_state[1]:.6f} rad/s")
print(f"Interpolated minimum steps to RoA = {estimated_steps}")


# FIND INITIAL CONDITION WITH AT LEAST 3 VIABLE STEPS
print("SELECTING INITIAL CONDITION REQUIRING AT LEAST 3 STEPS")
selected_index,selection_type=select_three_step_initial_condition(theta_dot_grid,steps_to_roa,maximum_steps_to_roa)

if selected_index is None:
    selected_initial_state=None
    print("No grid state has a viable route requiring or allowing at least 3 walking steps")
else:
    selected_initial_state=np.array([0.0,theta_dot_grid[selected_index]])
    print(f"Selected theta_dot = {selected_initial_state[1]:.6f} rad/s")
    print(f"Selection basis = {selection_type}")
    print(f"Minimum steps to RoA = {steps_to_roa[selected_index]}")
    print(f"Maximum viable steps to RoA = {maximum_steps_to_roa[selected_index]}")


# RUN MINIMUM-STEP POLICY FROM THE ORIGINAL INITIAL CONDITION
print("BEGINNING MINIMUM-STEP WALKING SIMULATION")
(state_history,alpha_history,poincare_history,completed_steps,reached_roa)=simulate_policy(initial_state,params,theta_dot_grid,alpha_grid,policy_alpha,steps_to_roa,maximum_steps_to_roa,max_walking_steps=max_walking_steps,timestep=timestep)

print("MINIMUM-STEP WALKING RESULT")
print(f"Number of walking steps = {completed_steps}")
print(f"Interpolated predicted steps = {estimated_steps}")
print(f"Reached balancing RoA = {reached_roa}")


# PLOT MINIMUM STEPS TO RoA
fig, ax=plt.subplots(figsize=(8,6),layout="constrained")
valid_steps=steps_to_roa >= 0
ax.plot(theta_dot_grid[valid_steps],steps_to_roa[valid_steps],marker="o",linestyle="-")
ax.axvline(initial_state[1],linestyle="--",label="Initial state")
ax.scatter(initial_state[1],estimated_steps,zorder=5)
ax.set_xlabel(r"$\dot{\theta}_k$ (rad/s)")
ax.set_ylabel("Minimum walking steps to RoA")
ax.set_title("Poincare States: Minimum Steps to Balancing RoA")
ax.legend()
fig.savefig(output_dir/"04_minimum_steps_to_roa.png",dpi=300)
plt.close(fig)


# PLOT MAXIMUM VIABLE STEPS
fig, ax=plt.subplots(figsize=(8,6),layout="constrained")
valid_maximum=maximum_steps_to_roa >= 0
ax.plot(theta_dot_grid[valid_maximum],maximum_steps_to_roa[valid_maximum],marker="o",linestyle="-")
ax.axhline(3,linestyle="--",label="3-step threshold")
ax.set_xlabel(r"$\dot{\theta}_k$ (rad/s)")
ax.set_ylabel("Maximum viable walking steps to RoA")
ax.set_title("Poincare States: Maximum Viable Steps to Balancing RoA")
ax.legend()
fig.savefig(output_dir/"05_maximum_steps_to_roa.png",dpi=300)
plt.close(fig)


# PLOT ORIGINAL WALKING TRAJECTORY
plot_poincare_trajectory(np.array(poincare_history),output_dir/"06_original_walking_poincare_trajectory.png","Poincare Walking Trajectory: Original Initial Condition")


# GENERATE ORIGINAL WALKING GIF
if len(state_history) > 1:
    generate_walking_gif(initial_state,alpha_history,params,output_dir/"07_original_walker.gif",timestep=timestep)

# Longest Viable Trajectory - still tweaking!
"""
# RUN LONGEST VIABLE TRAJECTORY
selected_history=[]
selected_alpha_history=[]
selected_poincare_history=[]
selected_completed_steps=0
selected_reached_roa=False

if selected_initial_state is not None:
    print("BEGINNING LONGEST VIABLE WALKING SIMULATION")
    (selected_history,selected_alpha_history,selected_poincare_history,selected_completed_steps,selected_reached_roa)=simulate_policy(selected_initial_state,params,theta_dot_grid,alpha_grid,policy_alpha,steps_to_roa,maximum_steps_to_roa,max_walking_steps=max_walking_steps,timestep=timestep,use_max_policy=True)

    print("LONGEST VIABLE WALKING RESULT")
    print(f"Number of walking steps = {selected_completed_steps}")
    print(f"Reached balancing RoA = {selected_reached_roa}")

    plot_poincare_trajectory(np.array(selected_poincare_history),output_dir/"08_three_step_walking_poincare_trajectory.png","Poincare Walking Trajectory: 3+ Step Initial Condition")
    generate_walking_gif(selected_initial_state,selected_alpha_history,params,output_dir/"09_three_step_walker.gif",timestep=timestep)

    #plot maximum-step trajectory separately
    fig, ax=plt.subplots(figsize=(8,6),layout="constrained")
    ax.plot(range(len(selected_poincare_history)),selected_poincare_history,marker="o")
    for k, velocity in enumerate(selected_poincare_history):
        ax.annotate(f"{velocity:.2f}",(k,velocity),xytext=(5,5),textcoords="offset points")
    ax.set_xlabel("Poincare step $k$")
    ax.set_ylabel(r"$\dot{\theta}_k$ (rad/s)")
    ax.set_title("Longest Viable Walking Trajectory Before RoA")
    fig.savefig(output_dir/"10_longest_viable_trajectory.png",dpi=300)
    plt.close(fig)
"""
