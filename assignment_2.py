from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
from integrators import rk4 as integrator
from integrators.rk4 import _rk4_step
from matplotlib.animation import FuncAnimation, PillowWriter
from models import inverted_pendulum_walker as model
from matplotlib.lines import Line2D


# CALCULATE CONTROLLER ROA IF A DATA FILE IS NOT ALREADY SAVED ###########################################################
roa_path = Path("output/assignment_2/roa.npz")
if roa_path.exists():
    print("Extracting ROA data...")
    roa_data = np.load(roa_path, allow_pickle=False)
    classification_grid = roa_data["classification_grid"]
    theta_values = roa_data["theta_values"]
    theta_dot_values = roa_data["theta_dot_values"]
else:
    print("No ROA data found, generating now...")
    params = model.generate_params()
    params["K_p"] = 10
    params["K_d"] = 10
    # Calculate ROA
    fig,ax,classification_grid, theta_values, theta_dot_values = model.plot_controller_roa(
        (params["incline"]-params["angle_of_attack"], params["incline"]+params["angle_of_attack"]),
        theta_dot_limits=(-1.5,1.5),
        n_theta=250,n_theta_dot=250, 
        params = params, 
        sim_time = 20, 
        timestep = 5e-4, 
        show=True)
    np.savez(roa_path,classification_grid=classification_grid,theta_values=theta_values,theta_dot_values=theta_dot_values)
    fig.savefig("output/assignment_2/controller_roa.png",dpi=300,bbox_inches="tight")

# Run a specific trial #######################################################################################################
params = model.generate_params()
timestep = 1e-3
sim_time = 5
sim_steps = int(sim_time / timestep)
state_traj = np.zeros((sim_steps+1,2))
x0 = [0,3]
state_traj[0,:] = x0
current_state = x0.copy()
time_traj = np.linspace(0,sim_time,sim_steps+1)
completed_steps = 0
print(f"Running simulation starting at {x0}")
first_roa = True
for step in range(sim_steps):
    if model.state_is_in_roa(current_state, theta_values, theta_dot_values, classification_grid):
        if first_roa:
            print(f"Reached the ROA:\n Seconds: {step*timestep:.3f}\n ROA State: {current_state}")
            first_roa = False
            ROA_state = current_state
        params["ankle_torque"] = model.calculate_torque(current_state,params)

    current_time = step * timestep
    next_state = _rk4_step(current_time,current_state,timestep,model.dynamics,params)

    if model.event_guard(current_state, next_state, params): # Collision occurs
        current_state = model.event_dynamics(next_state, params)
        state_traj[step+1, :] = current_state

        # Update Alpha
        completed_steps += 1
        new_angle_of_attack = np.pi/8
        #new_angle_of_attack = random.uniform(np.pi/8,np.pi/7)
        if new_angle_of_attack < np.pi/8 or new_angle_of_attack > np.pi/7:
            print("Angle of attack exceeded limits!")
            sys.exit()
        params["angle_of_attack"] = new_angle_of_attack
    else:
        current_state = next_state
        state_traj[step+1, :] = current_state

print("Simulation over, animating now...")
fig, ax = model.animate(state_traj.T,time_traj,params,timestep,completed_steps)
fig.savefig("output/assignment_2/animation.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

fig, ax = model.plot_phase_portrait(classification_grid, theta_values, theta_dot_values, state_traj, show=False)
fig.savefig("output/assignment_2/phase_portrait.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

# CREATE LOOKUP TABLE #######################################################################################################
base_params = model.generate_params()
gravity = base_params["gravity"]
length = base_params["length"]

# Discretization Parameters
n_theta_dot = 20
n_control_inputs = 3
control_inputs = np.linspace(np.pi / 8, np.pi / 7, n_control_inputs)
theta_dot_values_poincare = np.linspace(0, np.sqrt(2 * gravity / length), n_theta_dot)

lookup_table = np.full((n_theta_dot, n_control_inputs), np.nan)
reaches_roa = np.zeros((n_theta_dot, n_control_inputs), dtype=bool)
failed_transition = np.zeros((n_theta_dot, n_control_inputs),dtype=bool)

for velocity_index, theta_dot_0 in enumerate(theta_dot_values_poincare):
    for control_index, angle_of_attack in enumerate(control_inputs):
        # Simulate one poincare step i.e. until we hit theta = 0
        next_velocity, reached_roa, failed = model.simulate_poincare_step(
            theta_dot_0,
            angle_of_attack,
            base_params,
            roa_theta_values=theta_values,
            roa_theta_dot_values=theta_dot_values,
            classification_grid=classification_grid,
            timestep=1e-3,
            sim_time=5.0,
        )

        lookup_table[velocity_index, control_index] = next_velocity
        reaches_roa[velocity_index, control_index] = reached_roa
        failed_transition[velocity_index, control_index] = failed

print(f"Next-velocity lookup table:\n {lookup_table}")
print("Lookup table legend: \n -100 Indicates ROA is reached \n nan indicates the poincare section was not hit again \n All other numbers indicate the next velocity")

# Going from lookup table to paths #########################################################################################################################################

# First, we must go from our velocities and controls to our actual labels
def control_to_label(control, control_inputs):
    return int(np.argmin(np.abs(control_inputs - control)))

def velocity_to_label(theta_dot, theta_dot_values):
    return int(np.argmin(np.abs(theta_dot_values - theta_dot))) + 1

# This lookup table we now build uses the labels we have created, i.e. its the discretized node steps
lookup_table_w_labels = np.zeros_like(lookup_table)
for i,_ in enumerate(theta_dot_values_poincare):
    for j,__ in enumerate(control_inputs):
        if reaches_roa[i, j] or lookup_table[i,j] == -100:
            lookup_table_w_labels[i,j] = -100
        elif failed_transition[i, j] or np.isnan(lookup_table[i, j]):
            lookup_table_w_labels[i, j] = 0
        else:
            lookup_table_w_labels[i,j] = velocity_to_label(lookup_table[i,j],theta_dot_values_poincare)

print(f"\n Lookup table based on labels: \n {lookup_table_w_labels}")
print(f"Example: Having an entry of 10 in the [i,j] position means that from velocity label i, we can get to 10 via controller j")

# Next we build a recursive function to find all paths given our new lookup table with labels
def find_paths(depth, possible_steps, path, control_path, paths, control_paths):
    if depth == 0:
        paths.append(path)
        control_paths.append(control_path)
        return paths, control_paths

    if len(possible_steps) == 0:
        # Keep paths that end before reaching the requested depth.
        paths.append(path)
        control_paths.append(control_path)
        return paths, control_paths
    
    for idx in possible_steps:
        vel = idx[0]+1
        control = idx[1]
        new_path = path.copy()
        new_path.append(vel)
        new_control_path = control_path.copy()
        new_control_path.append(control)
        possible_steps = np.argwhere(lookup_table_w_labels == vel)
        find_paths(depth-1, possible_steps, new_path, new_control_path, paths, control_paths)

    return paths, control_paths

# Define dictionaries that contain all the paths of our robot
targets = {**{i: i for i in range(1, n_theta_dot + 1)},"GOAL": -100,"FAILURE": 0}
path_dict = {key: None for key in targets}
control_path_dict = {key: None for key in targets}

# MOST IMPORTANT PARAMETER IS DEPTH!!!! Tells us how many steps to backtrack from every point we start at within path building
depth = 5

# Iterate over recursive function that we built to find all paths
for target_name, table_value in targets.items():
    paths = []
    control_paths = []
    path = [target_name]
    control_path = []
    possible_steps = np.argwhere(lookup_table_w_labels == table_value)
    paths_for_dict, control_paths_for_dict = find_paths(depth, possible_steps, path, control_path, paths, control_paths)

    path_dict[target_name] = paths_for_dict
    control_path_dict[target_name] = control_paths_for_dict

# These dictionaries provide all possible paths we found. For example, path_dict[10] provides all paths starting at 10, and then backtrack to all other nodes that are connected

# Plotting Section ##################################################################################################
# Create labels
velocity = {
    node: float(value)
    for node, value in enumerate(theta_dot_values_poincare, start=1)
}
velocity_labels = {
    node: fr"{value:.2f}"
    for node, value in velocity.items()
}
control_labels = {
    index: f"{float(control):.4f} rad"
    for index, control in enumerate(control_inputs)
}
def nodes_in_paths(paths):
    return {
        node
        for path in paths
        for node in path
        if isinstance(node, (int, np.integer))
    }
goal_nodes = nodes_in_paths(path_dict["GOAL"])
failure_nodes = nodes_in_paths(path_dict["FAILURE"])
node_colors = {}
for node in range(1, n_theta_dot + 1):
    reaches_goal = node in goal_nodes
    reaches_failure = node in failure_nodes

    if reaches_goal and reaches_failure:
        node_colors[node] = "#D8B4FE"    # Purple: both
    elif reaches_goal:
        node_colors[node] = "#B7E4C7"    # Green: goal only
    elif reaches_failure:
        node_colors[node] = "#FFB3B3"    # Red: failure only
    else:
        node_colors[node] = "#E5E7EB"    # Gray: neither

# CALL PLOTTING FUNCTION
fig, ax, graph = model.plot_path_graph(
    path_dict,
    control_path_dict,
    keys=["GOAL", "FAILURE"],                 # Plot the complete table
    reverse=True,             # Important: these paths are already forward
    layout="spring", #"layout must be 'spring', 'kamada_kawai', 'circular', ""or 'hierarchical'."
    spring_spacing=2.2,
    control_labels=control_labels,
    spring_iterations=800,
    horizontal_spacing=6.0,  # Space between graph levels
    vertical_spacing=3.5,    # Space between nodes within each level
    node_labels=velocity_labels,
    node_label_mode="both",    # "id", "value", or "both"
    show_edge_labels=False,
    node_colors=node_colors,
    figsize=(14, 14),
    title=f"GOAL and FAILURE Reachability Graph\n num vel = {n_theta_dot}, num control = {n_control_inputs}",
)

# Create seperate node legend that we have to add onto it
# Save the control legend already created by plot_path_graph.
control_legend = ax.get_legend()

node_legend_handles = [
    Line2D(
        [], [],
        marker="o",
        linestyle="None",
        markerfacecolor="#B7E4C7",
        markeredgecolor="#333333",
        markersize=10,
        label="Can reach GOAL only",
    ),
    Line2D(
        [], [],
        marker="o",
        linestyle="None",
        markerfacecolor="#FFB3B3",
        markeredgecolor="#333333",
        markersize=10,
        label="Can reach FAILURE only",
    ),
    Line2D(
        [], [],
        marker="o",
        linestyle="None",
        markerfacecolor="#D8B4FE",
        markeredgecolor="#333333",
        markersize=10,
        label="Can reach both",
    ),
    Line2D(
        [], [],
        marker="o",
        linestyle="None",
        markerfacecolor="#E5E7EB",
        markeredgecolor="#333333",
        markersize=10,
        label="Can reach neither",
    ),
]

# Add the node-color legend below the control legend.
ax.legend(
    handles=node_legend_handles,
    title=f"Node reachability in {depth} steps",
    frameon=False,
    loc="lower left",
    bbox_to_anchor=(1.01, 0.0),
)

# Restore the original control legend so both remain visible.
if control_legend is not None:
    ax.add_artist(control_legend)

plt.show()

# Optional:
fig.savefig("output/assignment_2/paths_all_20_3.png", dpi=200, bbox_inches="tight")

sys.exit()
