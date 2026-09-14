import numpy as np


def dynamics(t, state, params):
    gravity = params["gravity"]
    k = params["k"]
    mass = params["mass"]
    damping_coeff = params["damping_coeff"]

    pos = state[0]
    velocity = state[1]

    acceleration = -gravity - (k/mass) * pos - (damping_coeff/mass * velocity)  # <-- DAMPING TERM

    state_derivative = np.array([velocity, acceleration])
    return state_derivative


def generate_params():
    params = {
        "gravity": 9.81,  # gravity m/s^2)
        "k": 1000,  # spring constant (N/m)
        "mass": 1,  # point mass at end of rod (kg)
        "damping_coeff": 0.1,  # damping coefficient (kg*m^2/s)
    }
    return params


def calculate_energy(state, params):
    """Compute energies for a state ``(2,)`` or trajectory ``(2, N)``."""
    gravity = params["gravity"]
    k = params["k"]
    mass = params["mass"]

    pos = state[0]  # indexes entire row "vectorized" if state is (2, N)
    velocity = state[1]

    kinetic_energy = 0.5 * mass * velocity ** 2
    potential_energy = mass*gravity*pos+0.5*k*pos**2
    return kinetic_energy, potential_energy