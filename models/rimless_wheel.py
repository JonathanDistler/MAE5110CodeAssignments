import numpy as np
from models import inverted_pendulum_walker as walker


#model of the pendulum, need to figure out how to add in the switch of coordinates
#need to figure out how to switch the coordinates
def dynamics(t, state, params):
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]
    damping_coeff = params["damping_coeff"]

    angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = (
        mass * gravity * length * np.sin(angle)
        - damping_coeff * angular_velocity  # <-- DAMPING TERM
    ) / (mass * length**2)

    state_derivative = np.array([angular_velocity, angular_acceleration])
    return state_derivative

def generate_initial_condition(params=None):
    params = generate_params() if params is None else params
    initial_angle = params["incline"]       # radians, centered over the ramp
    initial_velocity = 1.0    # radians per second
    state = np.array([initial_angle, initial_velocity])
    validate_initial_condition(state, params)
    return state


def generate_params():
    params = {
        "gravity": 9.81,  # gravity m/s^2)
        "length": 1,  # rod length (m)
        "mass": 1,  # point mass at end of rod (kg)
        "damping_coeff": 0,  # damping coefficient (kg*m^2/s)
        "N": 6, #number of spokes 
        "gamma": np.pi/6 #angle of incline
    }
    params["N_spokes"] = params["N"]
    params["incline"] = params["gamma"]
    return params

def calculate_positions(state, params):
    length = params["length"]
    angles = state[0] + np.arange(params["N_spokes"]) * 2 * np.pi / params["N_spokes"]
    hub = length * np.array([np.sin(state[0]), np.cos(state[0])])
    tips = hub - length * np.column_stack((np.sin(angles), np.cos(angles)))
    return hub, tips

def validate_initial_condition(state, params):
    state = np.asarray(state, dtype=float)
    if state.shape != (2,) or not np.all(np.isfinite(state)):
        raise ValueError("state must contain two finite values: [theta, velocity].")
    if params["N_spokes"] < 3 or int(params["N_spokes"]) != params["N_spokes"]:
        raise ValueError("N_spokes must be an integer of at least three.")
    if not np.isfinite(params["incline"]) or abs(params["incline"]) >= np.pi / 2:
        raise ValueError("incline must be finite and between -pi/2 and pi/2.")
    _, tips = calculate_positions(state, params)
    normal = np.array([np.sin(params["incline"]), np.cos(params["incline"])])
    if np.any(tips @ normal < -1e-12):
        raise ValueError("Initial angle places a leg below the ground plane.")

def event_guard(previous_state, next_state, params):
    collision_params = params.copy()
    collision_params["angle_of_attack"] = np.pi / params["N_spokes"]
    return walker.event_guard(previous_state, next_state, collision_params)

def event_dynamics(state, params):
    collision_params = params.copy()
    collision_params["angle_of_attack"] = np.pi / params["N_spokes"]
    return walker.event_dynamics(state, collision_params)

def advance_step(time, state, timestep, params):
    validate_initial_condition(state, params)
    collision_params = params.copy()
    collision_params["angle_of_attack"] = np.pi / params["N_spokes"]
    return walker.advance_step(time, state, timestep, collision_params, dynamics)

#need to figure out how to detect if the 2nd leg is touching
def is_touching(state, params):
    gamma=params["incline"]
    N=params["N_spokes"]
    two_alpha=(2*np.pi)/N
    alpha=two_alpha/2

    theta=state[0]

    return abs(theta-gamma)>=alpha


def reset_params(state,params):
    return event_dynamics(state, params)


def calculate_energy(state, params):
    """Compute energies for a state ``(2,)`` or trajectory ``(2, N)``."""
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    angle = state[0]  # indexes entire row "vectorized" if state is (2, N)
    angular_velocity = state[1]

    kinetic_energy = 0.5 * mass * (length * angular_velocity) ** 2
    potential_energy = mass * gravity * length * np.cos(angle)
    return kinetic_energy, potential_energy

def calculate_momentum(state,params):
    mass = params["mass"]
    length = params["length"]
    theta_dot=state[1]
    L=mass*length**2*theta_dot
    return(L)
