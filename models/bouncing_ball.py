import numpy as np


def dynamics(t, state, params):
    gravity = params["gravity"]
    mass = params["mass"]
    damping_coeff = params["damping_coeff"]

    height = state[0]
    velocity = state[1]

    acceleration = -gravity - damping_coeff * velocity / mass  # <-- DAMPING TERM

    state_derivative = np.array([velocity, acceleration])
    return state_derivative


def generate_params():
    params = {
        "gravity": 9.81,  # gravity m/s^2)
        "mass": 1,  # ball mass (kg)
        "damping_coeff": 0.0,  # air drag coefficient (kg/s)
        "restitution": 0.8,  # velocity fraction retained after a bounce, unused by dynamics()
    }
    return params


def calculate_energy(state, params):
    """Compute energies for a state ``(2,)`` or trajectory ``(2, N)``."""
    gravity = params["gravity"]
    mass = params["mass"]

    height = state[0]  # indexes entire row "vectorized" if state is (2, N)
    velocity = state[1]

    kinetic_energy = 0.5 * mass * velocity**2
    potential_energy = mass * gravity * height
    return kinetic_energy, potential_energy
