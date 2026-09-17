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
    #feedback-linearization controller, desired closed-loop dynamics are theta-ddot=-kp*theta-kd*theta; pretty much the superposition of drag and friction introduced


    length=params["length"]
    mass=params["mass"]
    gravity=params["gravity"]
    damping=params["damping_coeff"]

    theta=state[0]
    theta_dot=state[1]

    #future work could be done on optimizing kp*theta, kd*theta - considering using a Nelder-Mead optimization without any of the other simulation aspects 
    desired_acceleration=-kp*theta-kd*theta_dot
    required_torque=(mass*length**2*desired_acceleration - mass*gravity*length*np.sin(theta) + damping*theta_dot)

    #output is the required torque to produce intended dynamics
    return required_torque

# RoA
def in_RoA_helper(state, params, kp=2.0, kd=0.5):

    required_torque=feedback_linearization_controller(state, params, kp=kp, kd=kd)
    #gets the bound of torques, compares with the required torque 
    _, _, tau_min, tau_max=get_control_bounds(params)

    #state is inside the balancing RoA if the required controller torque is achievable
    return tau_min <= required_torque <= tau_max


# POINCARE SECTION
def poincare_section_event(prev_state, next_state, params):

    theta_prev=prev_state[0]
    theta_next=next_state[0]

    #defines the poincare section as passing through the veritcal upright position, walker leaves at theta<0 and towards theta>0
    return (theta_prev < 0 and theta_next >= 0)

#used to interpolate between grid points to find exact position of touchdown
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

    #ankle controller is off during walking as per the markdown 
    step_params["ankle_torque"]=0.0
    gamma=step_params["incline"]

    #touchdown angle for the inclined surface
    theta_td=gamma+alpha

    #start on the Poincare section
    state=np.array([0.0, theta_dot])

    #chosen to be arbitrarily high 
    max_steps=20000

    # FIND TOUCHDOWN
    touchdown_found=False

    for i in range(max_steps):

        next_state=(state + timestep*model.dynamics(0, state,step_params))

        if (state[0] < theta_td and next_state[0] >= theta_td):

            #interpolate to the exact touchdown angle
            touchdown_state=interpolate_state(state, next_state, theta_td)

            #apply impact dynamics
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

            #interpolate to theta=0
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
print()
print("BEGINNING BALANCING RoA SEARCH")


#small grid around the useful state range
theta_grid=np.linspace(-np.pi/2, np.pi/2, 81)
theta_dot_grid_roa=np.linspace(-6, 6, 121)

#only considering two options for RoA - in Roa or not in RoA
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

# PLOT BALANCING RoA
output_dir=Path("output/assignment_2")
output_dir.mkdir(parents=True, exist_ok=True)

fig, ax=plt.subplots(figsize=(8,6), layout="constrained")
ax.contourf(theta_grid, theta_dot_grid_roa, roa_grid.astype(float), levels=[0.5,1.5])
ax.axvline(0, linestyle="--")
ax.axhline(0, linestyle="--")
ax.set_xlabel(r"$\theta$ (rad)")
ax.set_ylabel(r"$\dot{\theta}$ (rad/s)")
ax.set_title("Balancing Controller Region of Attraction")
fig.savefig(output_dir / "balancing_roa.png", dpi=300)
plt.close(fig)


# RoA LOOKUP FUNCTION
def state_is_in_roa(state, params):

    #use the actual continuous state instead of the nearest RoA grid point, had been interplating from one step boundary to another (e.g. should've been 2, interpolated to 1)
    return in_RoA_helper(state, params, kp=kp, kd=kd)


def poincare_state_is_in_roa(theta_dot):
    state=np.array([0.0, theta_dot])
    return state_is_in_roa(state, params)


# POINCARE GRID SEARCH
print("BEGINNING POINCARE SEARCH")

#initial angular velocities
theta_dot_grid=np.linspace(0, np.sqrt(2*params["gravity"]/params["length"]), 41) #as per the Froude number

#allowable alpha values
alpha_grid=np.linspace(np.pi/8, np.pi/7, 21) #again, per the assignment 2 requirements


poincare_table=np.full((len(theta_dot_grid), len(alpha_grid)), np.nan)


for i, theta_dot in enumerate(theta_dot_grid):
    print(f"Searching theta_dot = {theta_dot:.4f} rad/s, {i+1}/{len(theta_dot_grid)}")

    for j, alpha in enumerate(alpha_grid):
        poincare_table[i,j]=poincare_map(theta_dot, alpha, params, timestep=timestep)


print("POINCARE SEARCH COMPLETE")

# POINCARE MAP SANITY CHECK
print("POINCARE MAP SANITY CHECK")

test_theta_dot_index=np.argmin(
    np.abs(theta_dot_grid - initial_state[1]))

for j, alpha in enumerate(alpha_grid):
    next_velocity=poincare_table[test_theta_dot_index, j]
    print(f"alpha = {alpha:.6f}, v_next = {next_velocity:.6f}")


# BACKWARD REACHABILITY
print("BEGINNING BACKWARD REACHABILITY")

#steps_to_roa[i] stores the minimum number of walking steps needed to reach the balancing RoA; 0 if in RoA, positive means steate can be reached, -1 if not reachable

steps_to_roa=np.full(len(theta_dot_grid),-1, dtype=int)

#first mark Poincare states that are already inside the balancing RoA
for i, theta_dot in enumerate(theta_dot_grid):
    if poincare_state_is_in_roa(theta_dot):
        steps_to_roa[i]=0


#store which alpha leads toward the RoA
policy_alpha=np.full(len(theta_dot_grid), np.nan)

#states already in the RoA do not need walking
for i in range(len(theta_dot_grid)):
    if steps_to_roa[i] == 0:
        policy_alpha[i]=alpha_grid[0]


#perform backward reachability
max_backward_iterations=len(theta_dot_grid)

for iteration in range(max_backward_iterations):

    previous_steps_to_roa=steps_to_roa.copy()
    for i, theta_dot in enumerate(theta_dot_grid):
        #skip states that are already reachable
        if previous_steps_to_roa[i] >= 0:
            continue

        for j, alpha in enumerate(alpha_grid):
            next_theta_dot=poincare_table[i,j]

            #invalid Poincare maps cannot be used
            if not np.isfinite(next_theta_dot):
                continue

            #the lookup table only contains positive walking velocities
            if next_theta_dot <= 0:
                continue

            #find the closest Poincare grid state
            next_index=np.argmin(
                np.abs(theta_dot_grid - next_theta_dot))

            #if the next grid state can reach the RoA, this state can also reach it
            if previous_steps_to_roa[next_index] >= 0:
                steps_to_roa[i]=(
                    previous_steps_to_roa[next_index] + 1)
                policy_alpha[i]=alpha
                break

    new_states=np.sum((steps_to_roa >= 0) & (previous_steps_to_roa < 0 ))

    print(f"Backward iteration {iteration+1}: {np.sum(steps_to_roa)>=0}, reachable states (+{new_states})")
    if np.array_equal(steps_to_roa, previous_steps_to_roa):
        break

print("BACKWARD REACHABILITY COMPLETE")
number_reachable=np.sum(steps_to_roa >= 0)
print(f"Reachable states: {number_reachable}/{len(steps_to_roa)}")

# LOOKUP POLICY
def lookup_alpha(theta_dot):
    #find closest Poincare grid point
    index=np.argmin(np.abs(theta_dot_grid - theta_dot))
    if steps_to_roa[index] < 0:

        return np.nan

    return policy_alpha[index]

# INTERPOLATE STEPS TO RoA
def interpolated_steps_to_roa(theta_dot):

    #if the state is already inside the balancing RoA, zero steps are needed
    if poincare_state_is_in_roa(theta_dot):
        return 0

    #states outside the lookup-table range cannot be interpolated
    if (theta_dot < theta_dot_grid[0] or theta_dot > theta_dot_grid[-1]):
        return -1

    #find the grid cell containing theta_dot
    upper_index=np.searchsorted(theta_dot_grid, theta_dot)

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

    #if either neighboring grid point is unreachable, we cannot interpolate through that cell
    if (lower_steps < 0 or upper_steps < 0):
        return -1

    #if both grid points are the same, there is no interpolation necessary
    if lower_index == upper_index:
        return int(lower_steps)

    lower_velocity=theta_dot_grid[lower_index]

    upper_velocity=theta_dot_grid[upper_index]
    fraction=(theta_dot - lower_velocity) / (upper_velocity - lower_velocity)
    interpolated_steps=(lower_steps + fraction*(upper_steps - lower_steps))

    #round to the nearest integer by adding 0.5 before casting
    integer_steps=int(interpolated_steps + 0.5)
    return integer_steps

# PRINT INTERPOLATED INITIAL STEPS
print("INTERPOLATED STEPS TO RoA")

estimated_steps=interpolated_steps_to_roa(initial_state[1])

print(f"Initial theta_dot = {initial_state[1]:.6f} rad/s")
print(f"Interpolated steps to RoA = {estimated_steps}")

# WALK ONE STEP
def integrate_to_touchdown(state, alpha, params, timestep=1e-4):
    step_params=params.copy()
    step_params["angle_of_attack"]=alpha

    #ankle controller remains off
    step_params["ankle_torque"]=0.0
    gamma=step_params["incline"]

    theta_td=gamma+alpha
    max_steps=20000 #again, arbitrarily high to test long term convergence

    for i in range(max_steps):
        next_state=(state + timestep*model.dynamics(0, state,step_params))

        if (state[0] < theta_td and next_state[0] >= theta_td):
            touchdown_state=interpolate_state(state, next_state, theta_td)
            return model.event_dynamics(touchdown_state, step_params)

        state=next_state

    return None


def integrate_to_poincare(state, params, timestep=1e-4):

    step_params=params.copy()
    step_params["ankle_torque"]=0.0
    max_steps=20000

    for i in range(max_steps):
        next_state=(state + timestep*model.dynamics(0, state, step_params))

        if poincare_section_event(state, next_state, step_params):
            section_state=interpolate_state(state, next_state, 0.0)
            return section_state

        state=next_state
    return None

# SIMULATE LOOKUP POLICY
def simulate_lookup_policy(initial_state, params, max_walking_steps=20, timestep=1e-4):

    state=initial_state.copy()
    state_history=[state.copy()]
    alpha_history=[]

    poincare_history=[state[1]]
    completed_steps=0

    for step in range(max_walking_steps):

        #check the actual continuous state against the balancing controller RoA
        if state_is_in_roa(state, params):
            print(f"Entered RoA after {completed_steps} walking steps")
            return (state_history, alpha_history, poincare_history, completed_steps, True)

        #Outside the RoA, so ankle torque remains off
        params["ankle_torque"]=0.0
        theta_dot=state[1]

        alpha=lookup_alpha(theta_dot)

        if not np.isfinite(alpha):
            print("Current state is not reachable by the lookup table")
            return (state_history, alpha_history, poincare_history, completed_steps, False)

        print(f"step {step+1}: alpha = {alpha:.5f}, theta_dot = {theta_dot:.5f}")

        alpha_history.append(alpha)

        #walk until touchdown
        touchdown_state=integrate_to_touchdown(state, alpha, params, timestep=timestep)

        if touchdown_state is None:
            print("Touchdown was not detected")

            return (state_history, alpha_history, poincare_history, completed_steps, False)

        #walk from touchdown until the
        #next Poincare section
        next_state=integrate_to_poincare(touchdown_state, params, timestep=timestep)

        if next_state is None:
            print("Next Poincare section was not detected")

            return (state_history,alpha_history, poincare_history,completed_steps,False)

        state=next_state.copy()
        state_history.append(state.copy())
        poincare_history.append(state[1])

        completed_steps += 1

        print(f"    next Poincare theta_dot = {state[1]:.5f}")

    print("Maximum walking steps reached without entering the RoA")

    return (state_history, alpha_history, poincare_history, completed_steps, False)

# FIND AN INITIAL CONDITION
print("SELECTING WALKING INITIAL CONDITION")

initial_theta_dot=initial_state[1]

initial_index=np.argmin(np.abs(theta_dot_grid - initial_theta_dot))

if steps_to_roa[initial_index] < 0:
    print(f"Initial theta_dot = {initial_theta_dot:.6f} rad/s is not reachable on the current grid")

else:
    print(f"theta_dot_0 = {initial_theta_dot:.6f} rad/s")
    print(f"Grid-predicted steps to RoA = {steps_to_roa[initial_index]}")
    print(f"Interpolated steps to RoA = {interpolated_steps_to_roa(initial_theta_dot)}")
    print(f"Selected alpha = {policy_alpha[initial_index]:.6f} rad")

# RUN WALKING SIMULATION
print("BEGINNING WALKING SIMULATION")

(state_history, alpha_history, poincare_history, completed_steps, reached_roa)=simulate_lookup_policy(initial_state, params, max_walking_steps=20, timestep=timestep)


# WALKING RESULT
print("WALKING RESULT")

print(f"Number of walking steps = {completed_steps}")
print(f"Interpolated predicted steps = {estimated_steps}")
print(f"Reached balancing RoA = {reached_roa}")

# PLOT STEPS TO RoA
fig, ax=plt.subplots(figsize=(8,6),layout="constrained")
valid_steps=steps_to_roa >= 0
ax.plot(theta_dot_grid[valid_steps], steps_to_roa[valid_steps], marker="o", linestyle="-")

#interpolated initial-condition estimate
ax.axvline(initial_state[1], linestyle="--", label="Initial state")
ax.scatter(initial_state[1], estimated_steps, zorder=5)
ax.set_xlabel(r"$\dot{\theta}_k$ (rad/s)")
ax.set_ylabel("Walking steps to RoA")
ax.set_title("Poincare States: Steps to Balancing RoA")
ax.legend()
fig.savefig(output_dir / "poincare_steps_to_roa.png", dpi=300)
plt.close(fig)

# PLOT POINCARE WALKING TRAJECTORY
fig, ax=plt.subplots(figsize=(8,6), layout="constrained")

#plot the balancing RoA on the Poincare section
poincare_roa=np.array([poincare_state_is_in_roa(v) for v in theta_dot_grid])

if np.any(poincare_roa):
    roa_velocities=theta_dot_grid[poincare_roa]
    ax.axhspan(np.min(roa_velocities), np.max(roa_velocities), alpha=0.15, label="Balancing RoA")

ax.plot(range(len(poincare_history)), poincare_history, marker="o")


#label the Poincare states
for k, velocity in enumerate(poincare_history):
    ax.annotate(f"{velocity:.2f}", (k, velocity), xytext=(5,5), textcoords="offset points")

ax.set_xlabel("Poincare step $k$")
ax.set_ylabel(r"$\dot{\theta}_k$ (rad/s)")
ax.set_title("Poincare Walking Trajectory")
ax.legend()
fig.savefig(output_dir / "walking_poincare_trajectory.png", dpi=300)
plt.close(fig)

# GENERATE WALKING GIF
if len(state_history) > 1:
    #create a full continuous simulation for visualization but only keep every 50th frame
    visualization_states=[]
    state=initial_state.copy()
    visualization_states.append(state.copy())

    #save one out of every 50 simulation frames
    frame_skip=50

    for step, alpha in enumerate(alpha_history):

        params["angle_of_attack"]=alpha
        params["ankle_torque"]=0.0

        #integrate through the stance phase
        gamma=params["incline"]
        theta_td=gamma+alpha

        max_steps=20000
        frame_counter=0

        for i in range(max_steps):
            next_state=(state + timestep*model.dynamics( 0, state, params))

            #only save every 50th frame
            if frame_counter % frame_skip == 0:
                visualization_states.append(next_state.copy())

            frame_counter += 1

            if (state[0] < theta_td and next_state[0] >= theta_td):
                touchdown_state=interpolate_state(state, next_state, theta_td)
                state=model.event_dynamics(touchdown_state, params)
                #always save the touchdown state
                visualization_states.append(state.copy())

                break

            state=next_state

    if len(visualization_states) > 1:
        fig, ax=plt.subplots( figsize=(8,6))

        def update(frame):
            ax.clear()
            model.visualize( visualization_states[frame],params, ax=ax)
            ax.set_title(f"Walking frame {frame+1}/{len(visualization_states)}")

        animation=FuncAnimation(fig, update, frames=len(visualization_states), interval=20)
        gif_path=(output_dir/"walker.gif")
        animation.save(gif_path, writer=PillowWriter(fps=30))

        plt.close(fig)
        print(f"Saved {gif_path} ({len(visualization_states)} frames")
else:
    print("No walking trajectory was generated, so the GIF was not created")


# GRID FIDELITY CHECK
print("GRID FIDELITY INFORMATION")

grid_spacing=(theta_dot_grid[1] - theta_dot_grid[0])
alpha_spacing=(alpha_grid[1] - alpha_grid[0])

print(f"Poincare velocity grid spacing = {grid_spacing:.6f} rad/s")
print(f"Alpha grid spacing = {alpha_spacing:.6f} rad")
print(f"Number of Poincare velocity states = {len(theta_dot_grid)}")
print(f"Number of alpha states = {len(alpha_grid)}")
print(f"Total Poincare state/action combinations = {poincare_table.size}")

print("The lookup table should be considered sufficiently resolved when increasing the grid resolution does not substantially change the predicted steps to the balancing RoA")