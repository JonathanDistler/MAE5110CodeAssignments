"""InvertedPendulumWalker starter model, with visualization provided.

Implement the model functions for Assignment 2. The visualizer works independently
of those functions; it draws a supplied state without advancing the simulation.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
import time
from matplotlib.patches import Patch
from integrators.rk4 import _rk4_step


def generate_params():
    """
    Generates useful parameters
    """
    params = {
        "gravity": 9.81,  # gravity m/s^2)
        "length": 1,  # rod length (m)
        "mass": 1,  # point mass at end of rod (kg)
        "incline": 0.06,
        "N_spokes": 10,
        "angle_of_attack": np.pi/8,
        "ankle_torque": 0,
        "K_p": 10,
        "K_d": 10
    }
    return params


def dynamics(t, state, params):
    """
    Calculates the dynamics x_dot = [theta_dot, theta_ddot] for a spokeless wheel
    
    args:
        t: time, ununsed since these dynamics are autonomous
        state: Contains the state x=[theta,theta_dot] where theta = 0 is the vertical axis
        params: useful parameters including gravity, length of the spokes, and angle of the axis (incline).

    Returns: 
        state_derivative: array of the derivative [theta_dot, theta double dot]
    """
    gravity = params["gravity"]
    length = params["length"]
    ankle_torque = params["ankle_torque"]
    mass = params["mass"]

    theta = state[0]       # Angle from world vertical
    theta_dot = state[1]

    theta_ddot = (gravity / length) * np.sin(theta) + ankle_torque / (mass * length**2)
    state_derivative = np.array([theta_dot, theta_ddot])
    return state_derivative

def event_guard(previous_state, next_state, params):
    incline = params["incline"]
    alpha = params["angle_of_attack"]

    previous_theta = previous_state[0]
    next_theta = next_state[0]
    next_theta_dot = next_state[1]

    # Express angles relative to the ramp normal
    previous_relative = previous_theta - incline
    next_relative = next_theta - incline

    forward_collision = (
        previous_relative < alpha <= next_relative
        and next_theta_dot > 0
    )

    backward_collision = (
        previous_relative > -alpha >= next_relative
        and next_theta_dot < 0
    )

    return forward_collision or backward_collision


def event_dynamics(state, params):
    theta = state[0]
    theta_dot = state[1]

    alpha = params["angle_of_attack"]

    if theta_dot > 0:
        # Forward collision
        theta -= 2 * alpha

    elif theta_dot < 0:
        # Backward collision
        theta += 2 * alpha

    theta_dot *= np.cos(2 * alpha)

    return np.array([theta, theta_dot])

def calculate_energy(state, params):
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    theta = state[0]
    theta_dot = state[1]

    kinetic_energy = 0.5 * mass * (length * theta_dot) ** 2
    potential_energy = mass * gravity * length * np.cos(theta)

    return kinetic_energy, potential_energy

def calculate_torque(state,params):
    mass = params["mass"]
    gravity = params["gravity"]
    length = params["length"]
    K_p = params["K_p"]
    K_d = params["K_d"]
    tau_lower_bound = -0.1*mass*gravity*length
    tau_upper_bound = 0.05*mass*gravity*length  
    new_ankle_torque = -mass*gravity*length*np.sin(state[0])-K_p*state[0]-K_d*state[1]
    return np.clip(new_ankle_torque,tau_lower_bound,tau_upper_bound)

def plot_controller_roa(theta_limits,theta_dot_limits,n_theta,n_theta_dot,params,sim_time,timestep,show=False):    
    BALANCED = 0
    FAILURE = 1
    INCONCLUSIVE = 2
    theta_tolerance=0.001
    velocity_tolerance=0.001

    theta_values = np.linspace(theta_limits[0], theta_limits[1], n_theta)
    theta_dot_values = np.linspace(theta_dot_limits[0], theta_dot_limits[1], n_theta_dot)

    classification_grid = np.full((n_theta_dot, n_theta),INCONCLUSIVE,dtype=int)

    sim_steps = int(sim_time / timestep)
    total_trajectories = n_theta * n_theta_dot
    completed_trajectories = 0
    start_time = time.perf_counter()

    for i, theta_dot_0 in enumerate(theta_dot_values):
        for j, theta_0 in enumerate(theta_values):
            params = generate_params()
            current_state = np.array([theta_0,theta_dot_0])
            outcome = INCONCLUSIVE

            for step in range(sim_steps):
                params["ankle_torque"] = calculate_torque(current_state,params)

                current_time = step * timestep
                next_state = _rk4_step(current_time,current_state,timestep,dynamics,params)

                if event_guard(current_state,next_state,params):
                    outcome = FAILURE
                    break

                current_state = next_state
                if abs(current_state[0]) <= theta_tolerance and abs(current_state[1]) <= velocity_tolerance:
                    outcome = BALANCED
                    break

            classification_grid[i,j] = outcome
            completed_trajectories += 1

        elapsed = time.perf_counter() - start_time
        average_time = elapsed / completed_trajectories
        remaining = total_trajectories - completed_trajectories
        eta = average_time * remaining
        percent = 100 * completed_trajectories / total_trajectories

        print(
            f"Completed {completed_trajectories}/{total_trajectories} "
            f"({percent:.1f}%) | Elapsed: {elapsed:.1f} s | "
            f"ETA: {eta:.1f} s",
            flush=True,
        )

    elapsed = time.perf_counter() - start_time
    print(f"RoA calculation completed in {elapsed:.1f} seconds.")

    colors = ["green", "orange", "lightgray"]
    labels = ["Captured", "Collision/failure", "Inconclusive"]

    fig, ax = plt.subplots(
        figsize=(9, 7),
        layout="constrained",
    )

    image = ax.imshow(
        classification_grid,
        origin="lower",
        extent=[
            theta_values[0],
            theta_values[-1],
            theta_dot_values[0],
            theta_dot_values[-1],
        ],
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(colors),
        vmin=-0.5,
        vmax=2.5,
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        ticks=[BALANCED, FAILURE, INCONCLUSIVE],
    )
    colorbar.ax.set_yticklabels(labels)

    ax.set_xlabel(r"Initial $\theta$ (rad)")
    ax.set_ylabel(r"Initial $\dot{\theta}$ (rad/s)")
    ax.set_title("Approximate region of attraction")

    if show:
        plt.show()

    return fig, ax, classification_grid,theta_values, theta_dot_values  

def state_is_in_roa(state, theta_values, theta_dot_values, classification_grid):
    theta, theta_dot = state

    # States outside the sampled grid are not considered inside the RoA.
    if not theta_values[0] <= theta <= theta_values[-1]:
        return False

    if not theta_dot_values[0] <= theta_dot <= theta_dot_values[-1]:
        return False

    theta_index = np.argmin(np.abs(theta_values - theta))
    theta_dot_index = np.argmin(np.abs(theta_dot_values - theta_dot))

    return (classification_grid[theta_dot_index, theta_index] == 0) 


def simulate_poincare_step(theta_dot_0,angle_of_attack,params,roa_theta_values,roa_theta_dot_values,classification_grid,timestep=1e-3,sim_time=5.0):
    '''
    Returns next_velocity, did we reach ROA, did we fail

    Although, if we just look at lookuptable we can tell nans are fails, -100 are ROA, and other numbers are velocities    
    '''
    params["angle_of_attack"] = angle_of_attack
    params["ankle_torque"] = 0.0

    current_state = np.array([0.0, theta_dot_0])
    sim_steps = int(sim_time / timestep)

    # This initial condition may already be inside the RoA.
    if state_is_in_roa(current_state,roa_theta_values,roa_theta_dot_values,classification_grid):
        return -100, True, False

    has_collided = False
    for step in range(sim_steps):
        current_time = step * timestep
        next_state = _rk4_step(current_time,current_state,timestep,dynamics,params)
        collision = event_guard(current_state,next_state,params)
        if collision:
            current_state = event_dynamics(next_state,params)
            has_collided = True
            if state_is_in_roa(current_state,roa_theta_values,roa_theta_dot_values,classification_grid):
                return -100, True, False
            continue

        # After one collision, the next negative-to-positive crossing is the next Poincare iterate.
        crossed_section = (has_collided and current_state[0] < 0 and next_state[0] >= 0 and next_state[1] > 0)

        if crossed_section:
            section_state = np.array([0.0, next_state[1]])
            reached_roa = state_is_in_roa(section_state,roa_theta_values,roa_theta_dot_values,classification_grid)
            return next_state[1], reached_roa, False

        current_state = next_state

        # The assignment says that reaching the RoA at any time is terminal.
        if state_is_in_roa(current_state,roa_theta_values,roa_theta_dot_values,classification_grid):
            return -100, True, False

    # No next section crossing and no RoA entry before the timeout. Thus, classify as a failed step
    return np.nan, False, True 

def plot_phase_portrait(classification_grid, theta_values, theta_dot_values, state_traj, show= False):
    balanced_region = classification_grid == 0
    fig, ax = plt.subplots()

    # Plot the balanced RoA with low opacity.
    ax.contourf(
        theta_values,
        theta_dot_values,
        balanced_region.astype(float),
        levels=[0.5, 1.5],
        colors=["royalblue"],
        alpha=0.25,
    )

    # Phase trajectory: theta versus theta_dot.
    trajectory, = ax.plot(
        state_traj[:, 0],
        state_traj[:, 1],
        color="black",
        label="State trajectory",
    )

    roa_legend = Patch(
        facecolor="royalblue",
        alpha=0.25,
        label="Balanced RoA",
    )

    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\dot{\theta}$ (rad/s)")
    ax.set_title("Phase Portrait")
    ax.legend(handles=[trajectory, roa_legend])
    ax.grid(alpha=0.2)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax

#####################################################################################################################

from collections import Counter, defaultdict
from math import log2, sqrt

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


def _token(value):
    """Create a hashable identity for scalar or array-like values."""
    if isinstance(value, np.generic):
        value = value.item()
    try:
        hash(value)
        return type(value).__qualname__, value
    except TypeError:
        return type(value).__qualname__, repr(value)


def _display(value, label_source):
    """Return (display_text, customized) for a mapping, callable, or None."""
    if label_source is None:
        return str(value), False
    if callable(label_source):
        try:
            return str(label_source(value)), True
        except (KeyError, IndexError, TypeError, ValueError):
            # A numeric-only formatter should gracefully fall back for terminal
            # nodes such as "ROA" and "FAILURE".
            return str(value), False
    try:
        if value in label_source:
            return str(label_source[value]), True
    except TypeError:
        pass
    return str(value), False


def _selected_keys(path_dict, keys):
    if keys is None:
        return list(path_dict)
    if isinstance(keys, (str, bytes)):
        return [keys]
    try:
        if keys in path_dict:
            return [keys]
    except TypeError:
        pass
    try:
        return list(keys)
    except TypeError:
        return [keys]


def _validate_records(path_dict, control_path_dict, keys):
    selected = _selected_keys(path_dict, keys)
    if not selected:
        raise ValueError("No dictionary keys were selected.")

    records = []
    for key in selected:
        if key not in path_dict:
            raise KeyError(f"{key!r} is not in path_dict.")
        if key not in control_path_dict:
            raise KeyError(f"{key!r} is not in control_path_dict.")

        paths = list(path_dict[key])
        control_paths = list(control_path_dict[key])
        if len(paths) != len(control_paths):
            raise ValueError(
                f"Key {key!r}: {len(paths)} paths but "
                f"{len(control_paths)} control paths."
            )

        for path_number, (path, controls) in enumerate(zip(paths, control_paths)):
            path = list(path)
            controls = list(controls)
            if not path:
                raise ValueError(f"Key {key!r}, path {path_number} is empty.")
            if len(controls) != len(path) - 1:
                raise ValueError(
                    f"Key {key!r}, path {path_number}: {len(path)} states "
                    f"require {len(path) - 1} controls, got {len(controls)}."
                )
            records.append((key, path, controls))

    if not records:
        raise ValueError("There are no paths to plot.")
    return records


def _state_sort_key(state):
    if isinstance(state, (int, float)):
        return 0, float(state)
    return 1, type(state).__qualname__, repr(state)


def _hierarchical_positions(
    simple_graph,
    multi_graph,
    horizontal_spacing,
    vertical_spacing,
):
    """
    Place directed edges left-to-right.

    Strongly connected components are collapsed only for layout calculation,
    so graphs containing cycles still work while every real node remains visible.
    """
    components = list(nx.strongly_connected_components(simple_graph))
    condensed = nx.condensation(simple_graph, scc=components)
    component_of = condensed.graph["mapping"]

    component_layer = {}
    for component in nx.topological_sort(condensed):
        predecessors = list(condensed.predecessors(component))
        component_layer[component] = (
            max(component_layer[parent] + 1 for parent in predecessors)
            if predecessors
            else 0
        )

    node_layer = {
        node: component_layer[component_of[node]]
        for node in simple_graph.nodes
    }
    nodes_by_layer = defaultdict(list)
    for node, layer in node_layer.items():
        nodes_by_layer[layer].append(node)

    max_layer = max(nodes_by_layer, default=0)
    for layer in range(max_layer + 1):
        nodes_by_layer[layer].sort(
            key=lambda node: _state_sort_key(multi_graph.nodes[node]["state"])
        )

    # Barycentric sweeps reduce crossings without requiring Graphviz.
    for _ in range(4):
        normalized_rank = {}
        for layer in range(max_layer + 1):
            layer_nodes = nodes_by_layer[layer]
            denominator = max(1, len(layer_nodes) - 1)
            normalized_rank.update(
                {
                    node: index / denominator
                    for index, node in enumerate(layer_nodes)
                }
            )

        for layer in range(1, max_layer + 1):
            old_order = {
                node: index for index, node in enumerate(nodes_by_layer[layer])
            }

            def predecessor_score(node):
                predecessors = [
                    parent
                    for parent in simple_graph.predecessors(node)
                    if node_layer[parent] < layer
                ]
                if not predecessors:
                    return float("inf"), old_order[node]
                return (
                    sum(normalized_rank[parent] for parent in predecessors)
                    / len(predecessors),
                    old_order[node],
                )

            nodes_by_layer[layer].sort(key=predecessor_score)

        normalized_rank = {}
        for layer in range(max_layer + 1):
            layer_nodes = nodes_by_layer[layer]
            denominator = max(1, len(layer_nodes) - 1)
            normalized_rank.update(
                {
                    node: index / denominator
                    for index, node in enumerate(layer_nodes)
                }
            )

        for layer in range(max_layer - 1, -1, -1):
            old_order = {
                node: index for index, node in enumerate(nodes_by_layer[layer])
            }

            def successor_score(node):
                successors = [
                    child
                    for child in simple_graph.successors(node)
                    if node_layer[child] > layer
                ]
                if not successors:
                    return float("inf"), old_order[node]
                return (
                    sum(normalized_rank[child] for child in successors)
                    / len(successors),
                    old_order[node],
                )

            nodes_by_layer[layer].sort(key=successor_score)

    positions = {}
    for layer in range(max_layer + 1):
        layer_nodes = nodes_by_layer[layer]
        midpoint = (len(layer_nodes) - 1) / 2
        for row, node in enumerate(layer_nodes):
            positions[node] = (
                layer * horizontal_spacing,
                (midpoint - row) * vertical_spacing,
            )

    return positions, nodes_by_layer


def _general_graph_positions(
    simple_graph,
    layout,
    layout_seed,
    spring_spacing,
    spring_iterations,
):
    """Return a non-layered layout based only on graph connectivity."""
    # Canonical insertion order makes a fixed seed reproducible even when the
    # input dictionaries or paths are supplied in a different order.
    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(sorted(simple_graph.nodes, key=repr))
    undirected_edges = set()
    for source, target in simple_graph.edges:
        if source == target:
            continue
        ordered_pair = tuple(sorted((source, target), key=repr))
        undirected_edges.add(ordered_pair)
    layout_graph.add_edges_from(sorted(undirected_edges, key=repr))
    number_of_nodes = len(layout_graph)
    if number_of_nodes == 1:
        only_node = next(iter(layout_graph))
        return {only_node: (0.0, 0.0)}

    if layout == "spring":
        # Larger spring_spacing increases the preferred node separation.
        preferred_distance = spring_spacing / sqrt(max(1, number_of_nodes))
        raw_positions = nx.spring_layout(
            layout_graph,
            seed=layout_seed,
            k=preferred_distance,
            iterations=spring_iterations,
            weight=None,
            scale=1.0,
        )
    elif layout == "kamada_kawai":
        if layout_graph.number_of_edges() == 0:
            raw_positions = nx.circular_layout(layout_graph, scale=1.0)
        else:
            raw_positions = nx.kamada_kawai_layout(
                layout_graph,
                weight=None,
                scale=1.0,
            )
    elif layout == "circular":
        raw_positions = nx.circular_layout(layout_graph, scale=1.0)
    else:
        raise ValueError(
            "layout must be 'spring', 'kamada_kawai', 'circular', "
            "or 'hierarchical'."
        )

    return {
        node: (float(position[0]), float(position[1]))
        for node, position in raw_positions.items()
    }


def plot_path_graph(
    path_dict,
    control_path_dict,
    keys=None,
    reverse=True,
    node_labels=None,
    show_node_ids=None,
    control_labels=None,
    show_edge_counts=True,
    cmap="turbo",
    node_size=1300,
    figsize=None,
    ax=None,
    merge_shared_nodes=True,
    node_label_mode=None,
    show_edge_labels=True,
    show_legend=True,
    horizontal_spacing=4.5,
    vertical_spacing=2.2,
    edge_label_font_size=8,
    title=None,
    layout="spring",
    layout_seed=42,
    spring_spacing=1.8,
    spring_iterations=600,
    count_repeated_edges=False,
    terminal_node_styles=None,
    node_colors=None
):
    """
    Plot selected path-dictionary entries as one directed graph.

    Shared state IDs are represented by one node across all selected keys when
    merge_shared_nodes=True. For example, if paths under keys 10, 11, and 12 all
    visit state 25, the figure contains one node 25 with all relevant edges.

    node_label_mode may be "id", "value", or "both". When omitted, the old
    behavior is preserved: mapped values are shown when node_labels is supplied;
    otherwise node IDs are shown. Explicit node_label_mode overrides the older
    show_node_ids option.

    Increase horizontal_spacing and vertical_spacing if the plot is still dense.
    Set show_edge_labels=False to keep colored edges and the legend while hiding
    inline control labels.
    layout="spring" creates a general force-directed graph based only on which
    states connect. Other options are "kamada_kawai", "circular", and the older
    left-to-right "hierarchical" layout. Increase spring_spacing for a looser
    spring graph. For hierarchical layout, use horizontal_spacing and
    vertical_spacing instead. Set show_edge_labels=False to keep colored edges
    and the legend while hiding inline control labels.

    count_repeated_edges=False treats repeated appearances of the same
    source/control/target transition as one physical graph edge. This is the
    recommended setting when paths were expanded from a transition table.

    terminal_node_styles=None applies default green/red styles to raw states
    "ROA" and "FAILURE". Pass {} to disable terminal styling or supply a
    mapping from terminal state to draw_networkx_nodes keyword overrides.

    Returns (fig, ax, graph).
    """
    records = _validate_records(path_dict, control_path_dict, keys)

    if node_colors is None:
        node_colors = {}

    if node_label_mode is None:
        if show_node_ids is not None:
            node_label_mode = (
                "both" if show_node_ids and node_labels is not None
                else "value" if node_labels is not None
                else "id"
            )
        else:
            node_label_mode = "value" if node_labels is not None else "id"

    if node_label_mode not in {"id", "value", "both"}:
        raise ValueError("node_label_mode must be 'id', 'value', or 'both'.")
    if horizontal_spacing <= 0 or vertical_spacing <= 0:
        raise ValueError("horizontal_spacing and vertical_spacing must be positive.")
    if spring_spacing <= 0:
        raise ValueError("spring_spacing must be positive.")
    if spring_iterations <= 0:
        raise ValueError("spring_iterations must be positive.")
    if layout not in {"spring", "kamada_kawai", "circular", "hierarchical"}:
        raise ValueError(
            "layout must be 'spring', 'kamada_kawai', 'circular', "
            "or 'hierarchical'."
        )

    if terminal_node_styles is None:
        terminal_node_styles = {
            "GOAL": {
                "node_color": "#CDEFD6",
                "edgecolors": "#2E7D32",
                "node_shape": "s",
                "node_size": 1.25 * node_size,
            },
            "FAILURE": {
                "node_color": "#FFD6D6",
                "edgecolors": "#B71C1C",
                "node_shape": "X",
                "node_size": 1.25 * node_size,
            },
        }
    elif terminal_node_styles is False:
        terminal_node_styles = {}
    else:
        terminal_node_styles = dict(terminal_node_styles)

    graph = nx.MultiDiGraph()
    edge_counts = defaultdict(Counter)
    raw_controls = {}

    for key, path, controls in records:
        path_nodes = []
        for step, state in enumerate(path):
            state_token = _token(state)
            node = (
                state_token
                if merge_shared_nodes
                else (_token(key), step, state_token)
            )
            if node not in graph:
                graph.add_node(
                    node,
                    state=state,
                    dictionary_keys=set(),
                    original_steps=set(),
                    visits=0,
                )
            graph.nodes[node]["dictionary_keys"].add(key)
            graph.nodes[node]["original_steps"].add(step)
            graph.nodes[node]["visits"] += 1
            path_nodes.append(node)

        for step, control in enumerate(controls):
            control_token = _token(control)
            raw_controls.setdefault(control_token, control)
            if reverse:
                source, target = path_nodes[step + 1], path_nodes[step]
            else:
                source, target = path_nodes[step], path_nodes[step + 1]
            if count_repeated_edges:
                edge_counts[source, target][control_token] += 1
            else:
                edge_counts[source, target][control_token] = 1

    for (source, target), counts in edge_counts.items():
        for control_token, count in counts.items():
            graph.add_edge(
                source,
                target,
                key=control_token,
                control=raw_controls[control_token],
                count=count,
            )

    simple_graph = nx.DiGraph()
    simple_graph.add_nodes_from(graph.nodes)
    simple_graph.add_edges_from(
        (source, target)
        for source, target in edge_counts
        if source != target
    )
    if layout == "hierarchical":
        positions, nodes_by_layer = _hierarchical_positions(
            simple_graph,
            graph,
            horizontal_spacing,
            vertical_spacing,
        )
    else:
        positions = _general_graph_positions(
            simple_graph,
            layout,
            layout_seed,
            spring_spacing,
            spring_iterations,
        )
        nodes_by_layer = None
    graph.graph["positions"] = positions
    graph.graph["layout"] = layout

    control_tokens = sorted(raw_controls, key=repr)
    color_map = plt.get_cmap(cmap)
    if len(control_tokens) == 1:
        colors = {control_tokens[0]: color_map(0.5)}
    else:
        colors = {
            token: color_map(0.05 + 0.90 * index / (len(control_tokens) - 1))
            for index, token in enumerate(control_tokens)
        }

    if ax is None:
        if figsize is None:
            if layout == "hierarchical":
                number_of_layers = max(1, len(nodes_by_layer))
                largest_layer = max(
                    (len(nodes) for nodes in nodes_by_layer.values()),
                    default=1,
                )
                figsize = (
                    max(
                        11,
                        2.5
                        + 0.95 * horizontal_spacing * (number_of_layers - 1),
                    ),
                    max(
                        6.5,
                        3.0 + 0.45 * vertical_spacing * (largest_layer - 1),
                    ),
                )
            else:
                side = max(
                    9.0,
                    4.0 + 1.1 * sqrt(max(1, len(graph))) * spring_spacing,
                )
                figsize = (side, side)
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    def node_text(state):
        value_text, customized = _display(state, node_labels)
        if node_label_mode == "id":
            return str(state)
        if node_label_mode == "value":
            return value_text if customized else str(state)
        return f"{state}\n{value_text}" if customized else str(state)

    for node, attributes in graph.nodes(data=True):
        attributes["display_label"] = node_text(attributes["state"])
        attributes["is_terminal"] = False
        attributes["terminal_kind"] = None

    terminal_nodes = set()
    node_sizes = {node: float(node_size) for node in graph.nodes}
    for terminal_state, style in terminal_node_styles.items():
        nodelist = [
            node
            for node, data in graph.nodes(data=True)
            if data["state"] == terminal_state
        ]
        if not nodelist:
            continue
        terminal_nodes.update(nodelist)
        terminal_size = float(style.get("node_size", 1.25 * node_size))
        for node in nodelist:
            graph.nodes[node]["is_terminal"] = True
            graph.nodes[node]["terminal_kind"] = terminal_state
            node_sizes[node] = terminal_size
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=ax,
            nodelist=nodelist,
            node_size=terminal_size,
            node_color=style.get("node_color", "#F7F7F7"),
            edgecolors=style.get("edgecolors", "#333333"),
            linewidths=style.get("linewidths", 1.8),
            node_shape=style.get("node_shape", "s"),
        )

    ordinary_nodes = [node for node in graph.nodes if node not in terminal_nodes]
    if ordinary_nodes:
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=ax,
            node_size=node_size,
            nodelist = ordinary_nodes,
            node_color=[
            node_colors.get(
                graph.nodes[node]["state"],
                "#F7F7F7",
            )
            for node in ordinary_nodes
        ],
            edgecolors="#333333",
            linewidths=1.2,
        )
    nx.draw_networkx_labels(
        graph,
        positions,
        ax=ax,
        labels={
            node: data["display_label"]
            for node, data in graph.nodes(data=True)
        },
        font_size=9,
    )

    node_shrink = max(12, sqrt(node_size) / 2)
    x_values = [position[0] for position in positions.values()]
    y_values = [position[1] for position in positions.values()]
    position_span = max(
        max(x_values) - min(x_values) if x_values else 0.0,
        max(y_values) - min(y_values) if y_values else 0.0,
        1.0,
    )
    for (source, target), counts in edge_counts.items():
        items = sorted(counts.items(), key=lambda item: repr(item[0]))
        reverse_pair_exists = source != target and (target, source) in edge_counts
        if reverse_pair_exists:
            base_curve = 0.14 if repr(source) < repr(target) else -0.14
        else:
            base_curve = 0.0
        # The same positive curvature in both directions puts reciprocal
        # arrows on opposite physical sides because their endpoints reverse.
        base_curve = 0.14 if reverse_pair_exists else 0.0

        if len(items) == 1:
            curvatures = [base_curve]
        else:
            spread = min(0.42, 0.11 * (len(items) - 1))
            curvatures = [
                base_curve - spread + 2 * spread * index / (len(items) - 1)
                for index in range(len(items))
            ]

        x1, y1 = positions[source]
        x2, y2 = positions[target]
        for curvature, (control_token, count) in zip(curvatures, items):
            color = colors[control_token]
            source_shrink = max(12, sqrt(node_sizes[source]) / 2)
            target_shrink = max(12, sqrt(node_sizes[target]) / 2)

            if source == target:
                nx.draw_networkx_edges(
                    graph,
                    positions,
                    ax=ax,
                    edgelist=[(source, target, control_token)],
                    edge_color=[color],
                    width=1.5 + 0.8 * log2(count),
                    arrows=True,
                    arrowsize=15,
                    node_size=node_sizes[source],
                    connectionstyle="arc3,rad=0.45",
                )
                label_x = x1 + 0.08 * position_span
                label_y = y1 + 0.08 * position_span
            else:
                ax.add_patch(
                    FancyArrowPatch(
                        (x1, y1),
                        (x2, y2),
                        arrowstyle="-|>",
                        mutation_scale=15,
                        connectionstyle=f"arc3,rad={curvature}",
                        color=color,
                        linewidth=1.5 + 0.8 * log2(count),
                        shrinkA=source_shrink,
                        shrinkB=target_shrink,
                        zorder=1,
                    )
                )
                label_x = (x1 + x2) / 2 + 0.45 * curvature * (y2 - y1)
                label_y = (y1 + y2) / 2 - 0.45 * curvature * (x2 - x1)

            if show_edge_labels:
                control_text, _ = _display(
                    raw_controls[control_token],
                    control_labels,
                )
                text = f"u={control_text}"
                if show_edge_counts and count > 1:
                    text += f" ×{count}"
                ax.text(
                    label_x,
                    label_y,
                    text,
                    ha="center",
                    va="center",
                    fontsize=edge_label_font_size,
                    color=color,
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.88,
                    },
                    zorder=3,
                )

    if show_legend and control_tokens:
        handles = []
        for control_token in control_tokens:
            control_text, _ = _display(
                raw_controls[control_token],
                control_labels,
            )
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=colors[control_token],
                    linewidth=3,
                    label=f"u={control_text}",
                )
            )
        ax.legend(
            handles=handles,
            title="Control",
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
        )

    ax.set_title(
        title
        if title is not None else ("Merged paths in reverse order" if reverse else "Merged paths") 
    )
    ax.set_xlabel("Path direction  →")
    ax.set_xlabel("Path direction  →" if layout == "hierarchical" else "")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.margins(x=0.16, y=0.18)

    return fig, ax, graph


########################################################
def visualize(
    state,
    params,
    ax=None,
    *,
    show_swing=True,
    stance_position=(0.0, 0.0),
    view_limits=None,
):
    """Draw one walker pose and return a Matplotlib Axes.

    Parameters
    ----------
    state : array-like, shape (2,)
        [theta, angular_velocity], in radians and radians/second. Theta is
        measured clockwise from upward vertical; positive x points right.
    params : dict
        ``length`` is the leg length in meters. ``incline`` is the ground's
        downhill slope angle in radians (positive slopes descend to the right).
        ``angle_of_attack`` is HALF the angle between the stance and forward swing
        legs, in radians; it is needed only when show_swing=True.
        ``ankle_torque`` (optional, default 0) is displayed in N m, with positive
        torque acting in the positive theta direction. Other keys are ignored.
    ax : matplotlib.axes.Axes, optional
        Axes to clear and reuse. If omitted, create a figure. This function
        neither shows nor saves it: use plt.show() or ax.figure.savefig(...).
    show_swing : bool
        Draw a straight forward swing leg at the supplied angle_of_attack. Set False
        while the swing leg is held clear or while balancing. Swing motion is
        not part of the two-state model and is not inferred from theta.
    stance_position : pair of floats
        Current stance foot's (x, y) in meters, default (0, 0). The two-state
        model does not track translation; supply foot positions if desired.
        Ground passes through this point at the supplied incline.
    view_limits : (xmin, xmax, ymin, ymax), optional
        Fixed camera bounds in meters. By default the view follows the stance
        foot with bounds that fit both legs at any angle. Supply the same bounds
        each frame for a stationary world view.

    Notes
    -----
    Draws the supplied pose; contact events belong in the simulation.
    Reuse ax for frame sequences; use evenly spaced simulation times for playback
    at a fixed frame rate, and pass the parameters actually used at each frame.
    """
    state = np.asarray(state, dtype=float)
    foot = np.asarray(stance_position, dtype=float)
    if state.shape != (2,) or not np.all(np.isfinite(state)):
        raise ValueError("state must contain two finite values: [theta, velocity].")
    if foot.shape != (2,) or not np.all(np.isfinite(foot)):
        raise ValueError("stance_position must contain two finite values: [x, y].")
    length = float(params["length"])
    incline = float(params["incline"])
    torque = float(params.get("ankle_torque", 0.0))
    if not np.isfinite(length) or length <= 0:
        raise ValueError("length must be finite and positive.")
    if not np.isfinite(incline) or abs(incline) >= np.pi / 2:
        raise ValueError("incline must be finite and between -pi/2 and pi/2.")
    if not np.isfinite(torque):
        raise ValueError("ankle_torque must be finite.")
    if show_swing:
        angle_of_attack = float(params["angle_of_attack"])
        if not np.isfinite(angle_of_attack):
            raise ValueError("angle_of_attack must be finite.")

    if view_limits is None:
        radius = 2.15 * length
        view_limits = (
            foot[0] - radius,
            foot[0] + radius,
            foot[1] - radius,
            foot[1] + radius,
        )
    limits = np.asarray(view_limits, dtype=float)
    if (
        limits.shape != (4,)
        or not np.all(np.isfinite(limits))
        or limits[0] >= limits[1]
        or limits[2] >= limits[3]
    ):
        raise ValueError(
            "view_limits must be (xmin, xmax, ymin, ymax) with increasing bounds."
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    ax.clear()
    theta, angular_velocity = state
    hub = foot + length * np.array([np.sin(theta), np.cos(theta)])

    ground_x = np.array(limits[:2])
    ground_y = foot[1] - np.tan(incline) * (ground_x - foot[0])
    ax.fill_between(ground_x, ground_y, limits[2], color="#eee7dc", zorder=0)
    ax.plot(ground_x, ground_y, color="#7b6651", linewidth=2, label="Ground")
    ax.plot(
        [foot[0], foot[0]],
        [foot[1], foot[1] + 1.25 * length],
        ":",
        color="0.7",
        linewidth=1,
        label="Vertical",
    )

    if show_swing:
        swing_angle = theta - 2 * angle_of_attack
        swing_foot = hub - length * np.array([np.sin(swing_angle), np.cos(swing_angle)])
        swing_color = "#df8a25"
        ax.plot(
            [hub[0], swing_foot[0]],
            [hub[1], swing_foot[1]],
            "--",
            color=swing_color,
            linewidth=2.5,
            label="Swing leg",
            zorder=3,
        )
        ax.plot(
            *swing_foot,
            "o",
            color=swing_color,
            markersize=7,
            zorder=4,
            label="Swing foot",
        )

    stance_color = "#23699b"
    ax.plot(
        [foot[0], hub[0]],
        [foot[1], hub[1]],
        color=stance_color,
        linewidth=4,
        label="Stance leg",
        zorder=4,
    )
    ax.plot(*foot, "s", color="#333333", markersize=8, zorder=5, label="Stance foot")
    ax.plot(
        *hub,
        "o",
        color=stance_color,
        markeredgecolor="white",
        markersize=17,
        zorder=6,
        label="Hub",
    )
    ax.text(
        0.03,
        0.97,
        f"$\\theta$ = {theta:.3f} rad\n"
        f"$\\dot\\theta$ = {angular_velocity:.3f} rad/s\n"
        f"$\\tau$ = {torque:.3f} N m",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    ax.set(
        xlim=limits[:2],
        ylim=limits[2:],
        xlabel="x (m)",
        ylabel="y (m)",
        title="Inverted pendulum walker",
    )
    ax.set_aspect("equal", adjustable="box")
    return ax


def animate(state_traj,time_traj,params,timestep,completed_steps):
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")


    def draw_frame(index):
        # The massless swing leg is repositioned instantaneously at each impact.
        visualize(state_traj[:, index], params, ax=ax)
        ax.set_title(f"t = {time_traj[index]:.2f} s")


    # Simulate at a small timestep, but render only 25 frames per second.
    fps = 25
    frame_stride = round(1 / (fps * timestep))
    frame_indices = list(range(0, time_traj.size, frame_stride))
    if frame_indices[-1] != time_traj.size - 1:
        frame_indices.append(time_traj.size - 1)

    animation = FuncAnimation(
        fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
    )
    output = Path("output/assignment_2")
    output.mkdir(parents=True, exist_ok=True)
    animation.save(output / "walker.gif", writer=PillowWriter(fps=fps))

    # To save an MP4 instead, install FFmpeg and use:
    # animation.save(output / "walker.mp4", writer="ffmpeg", fps=fps)
    print(f"Saved {output / 'walker.gif'} ({completed_steps} footstrikes).")
    plt.show()

    return fig, ax
