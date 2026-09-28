import numpy as np
import matplotlib.pyplot as plt

from models import rimless_wheel as model
from wheel_integrator import rk4 as integrate


# PARAMETERS
params = {
    "gravity": 9.81,
    "length": 1.0,
    "mass": 1.0,
    "damping_coeff": 0.0,
    "N": 6,
    "gamma": np.pi / 6
}

sim_time = 20
timestep = 0.001


# BASIC GEOMETRY
def get_alpha(params):
    # Half the angle between neighboring spokes
    return np.pi / params["N"]
 
# FIXED POINT
def calculate_fixed_point(params):
   # Calculate the post-impact angular velocity of the steady rolling limit cycle

    g = params["gravity"]
    length = params["length"]
    N = params["N"]
    gamma = params["gamma"]

    alpha = np.pi / N

    # Velocity-squared gained during the stance phase
    velocity_squared_gain = (
        4 * g / length * np.sin(alpha) * np.sin(gamma)
    )

    # Velocity loss at impact
    impact_factor = np.cos(2 * alpha)

    if gamma <= 0:
        return np.nan

    omega_squared = (
        impact_factor**2 * velocity_squared_gain
        / (1 - impact_factor**2)
    )

    return np.sqrt(omega_squared)
 
# FORWARD WALKING THRESHOLD
def calculate_walking_threshold(params):

    g = params["gravity"]
    length = params["length"]
    N = params["N"]
    gamma = params["gamma"]

    alpha = np.pi / N

    threshold_squared = (
        2 * g / length * (1 - np.cos(gamma - alpha))
    )

    return np.sqrt(max(0, threshold_squared))
 
# RUN ONE STEP
def run_one_step(omega, params, timestep, sim_time):

    alpha = get_alpha(params)
    gamma = params["gamma"]

    # Start immediately after impact
    initial_state = np.array([
        gamma - alpha,
        omega
    ])

    return integrate(
        model.dynamics,
        model.is_touching,
        model.reset_params,
        initial_state,
        timestep,
        sim_time,
        params,
        num_contacts=1
    )

# LIMIT CYCLE
def calculate_limit_cycle(params, timestep, sim_time):

    omega_fixed = calculate_fixed_point(params)

    if not np.isfinite(omega_fixed):
        return None, None, np.nan

    time, state, theta_dot, contacts, _, _ = run_one_step(
        omega_fixed,
        params,
        timestep,
        sim_time
    )

    # If the wheel simply settles without hitting the next spoke, there is no rolling limit cycle
    if len(contacts) == 0:
        return None, None, omega_fixed

    contact = contacts[0]

    return (
        time[:contact],
        state[:, :contact],
        omega_fixed
    )

# RETURN MAP
def return_map(omega, params, timestep, sim_time):

    _, _, theta_dot, _, _, _ = run_one_step(
        omega,
        params,
        timestep,
        sim_time
    )

    if len(theta_dot) == 0:
        return np.nan

    # Angular velocity immediately after the next impact
    return theta_dot[0]


def calculate_return_map(
    params,
    timestep,
    sim_time,
    omega_min=0.1,
    omega_max=3.5,
    num_points=100
):

    omega_values = np.linspace(
        omega_min,
        omega_max,
        num_points
    )

    next_omega = np.array([
        return_map(omega, params, timestep, sim_time)
        for omega in omega_values
    ])

    valid = np.isfinite(next_omega)

    return omega_values[valid], next_omega[valid]
 
# FLOQUET MULTIPLIER
def calculate_floquet_multiplier(
    params,
    timestep,
    sim_time,
    delta=0.05
):

    omega_fixed = calculate_fixed_point(params)

    if not np.isfinite(omega_fixed):
        return np.nan

    omega_plus = return_map(
        omega_fixed + delta,
        params,
        timestep,
        sim_time
    )

    omega_minus = return_map(
        omega_fixed - delta,
        params,
        timestep,
        sim_time
    )

    if not (
        np.isfinite(omega_plus)
        and np.isfinite(omega_minus)
    ):
        return np.nan

    # Numerical derivative of the return map
    return (omega_plus - omega_minus) / (2 * delta)

# CONVERGENCE
def calculate_convergence(
    params,
    timestep,
    sim_time,
    initial_omega,
    num_contacts=10
):

    alpha = get_alpha(params)
    gamma = params["gamma"]

    initial_state = np.array([
        gamma - alpha,
        initial_omega
    ])

    _, _, theta_dot, _, _, _ = integrate(
        model.dynamics,
        model.is_touching,
        model.reset_params,
        initial_state,
        timestep,
        sim_time,
        params,
        num_contacts=num_contacts
    )

    return np.array(theta_dot)

# CLASSIFY AN INITIAL CONDITION
def classify_initial_condition(
    initial_state,
    params,
    timestep,
    sim_time,
    tolerance=0.02
):

    _, _, theta_dot, _, _, _ = integrate(
        model.dynamics,
        model.is_touching,
        model.reset_params,
        initial_state,
        timestep,
        sim_time,
        params,
        num_contacts=None
    )

    omega_fixed = calculate_fixed_point(params)

    # Check whether the wheel approaches the rolling fixed point
    if (
        len(theta_dot) >= 5
        and np.isfinite(omega_fixed)
    ):

        recent = np.array(theta_dot[-5:])

        if np.all(
            np.abs(recent - omega_fixed) < tolerance
        ):
            return 1

    # Otherwise classify it as non-walking/falling
    return 0


# REGION OF ATTRACTION
def calculate_roa(
    params,
    timestep,
    sim_time,
    theta_values,
    omega_values
):

    roa = np.zeros(
        (len(omega_values), len(theta_values))
    )

    for i, omega in enumerate(omega_values):

        print(
            f"RoA row {i + 1}/{len(omega_values)}"
        )

        for j, theta in enumerate(theta_values):

            initial_state = np.array([
                theta,
                omega
            ])

            roa[i, j] = classify_initial_condition(
                initial_state,
                params,
                timestep,
                sim_time
            )

    return roa


def calculate_roa_fraction(
    params,
    timestep,
    sim_time,
    num_points=35
):

    gamma = params["gamma"]
    alpha = get_alpha(params)

    theta_values = np.linspace(
        gamma - alpha,
        gamma + alpha,
        num_points
    )

    omega_fixed = calculate_fixed_point(params)

    if np.isfinite(omega_fixed):
        omega_max = max(3.0, 1.5 * omega_fixed)
    else:
        omega_max = 3.0

    omega_values = np.linspace(
        0,
        omega_max,
        num_points
    )

    roa = calculate_roa(
        params,
        timestep,
        sim_time,
        theta_values,
        omega_values
    )

    return np.mean(roa == 1)
 
# PLOT REGION OF ATTRACTION
def plot_roa(
    theta_values,
    omega_values,
    roa,
    params,
    limit_cycle=None
):

    gamma = params["gamma"]

    plt.figure()

    plt.pcolormesh(
        theta_values,
        omega_values,
        roa,
        shading="auto"
    )

    if limit_cycle is not None:
        plt.plot(
            limit_cycle[0],
            limit_cycle[1],
            linewidth=2,
            label="Rolling limit cycle"
        )

        plt.scatter(
            limit_cycle[0, 0],
            limit_cycle[1, 0],
            s=70,
            label="Post-impact fixed point"
        )

        plt.legend()

    plt.xlabel(r"$\theta$ [rad]")
    plt.ylabel(r"$\dot{\theta}$ [rad/s]")
    plt.title(
        f"Region of Attraction, "
        f"N = {params['N']}, "
        f"$\\gamma$ = {np.degrees(gamma):.1f}^\\circ$"
    )
    
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "assignment_1_graphs/RoA_N6_g30_2.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

# PARAMETER SWEEPS
def run_sweep(params, values, parameter, timestep, sim_time):

    roa_values = []
    floquet_values = []

    for value in values:

        print(
            f"{parameter.upper()} = {value}"
        )

        sweep_params = params.copy()
        sweep_params[parameter] = value

        floquet = calculate_floquet_multiplier(
            sweep_params,
            timestep,
            sim_time
        )

        roa = calculate_roa_fraction(
            sweep_params,
            timestep,
            sim_time
        )

        roa_values.append(roa)
        floquet_values.append(floquet)

    return (
        np.array(roa_values),
        np.array(floquet_values)
    )


# PLOT PARAMETER SWEEP
def plot_sweep(
    x_values,
    roa_values,
    floquet_values,
    xlabel,
    roa_filename,
    floquet_filename
):

    # REGION OF ATTRACTION
    plt.figure()

    plt.plot(
        x_values,
        roa_values,
        marker="o",
        label="Rolling limit-cycle RoA"
    )

    plt.xlabel(xlabel)
    plt.ylabel("Fraction of sampled state space")
    plt.title("RoA vs Parameter")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        roa_filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # FLOQUET MULTIPLIER
    plt.figure()

    plt.plot(
        x_values,
        np.abs(floquet_values),
        marker="o",
        label=r"$|\lambda|$"
    )

    plt.axhline(
        1,
        linestyle="--",
        label="Stability boundary"
    )

    plt.xlabel(xlabel)
    plt.ylabel(r"$\lambda$")
    plt.title("Floquet Multiplier vs Parameter")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        floquet_filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# MAIN ANALYSIS
if __name__ == "__main__":

    # BASIC GEOMETRY
    N = params["N"]
    gamma = params["gamma"]
    alpha = get_alpha(params)

    # FIXED POINT AND WALKING THRESHOLD
    omega_fixed = calculate_fixed_point(params)
    walking_threshold = calculate_walking_threshold(params)

    print("\nLIMIT CYCLE")
    print(f"Fixed point velocity = {omega_fixed:.6f} rad/s")
    print(
        f"Forward walking threshold = "
        f"{walking_threshold:.6f} rad/s"
    )
    print(
        f"Post-impact theta = "
        f"{gamma - alpha:.6f} rad"
    )
    print(
        f"Pre-impact theta = "
        f"{gamma + alpha:.6f} rad"
    )
 
    # LIMIT CYCLE
    time_cycle, limit_cycle, _ = calculate_limit_cycle(
        params,
        timestep,
        sim_time
    )

    if limit_cycle is not None:

        print(
            f"Number of points = "
            f"{limit_cycle.shape[1]}"
        )

        plt.figure()

        plt.plot(
            limit_cycle[0],
            limit_cycle[1],
            linewidth=2,
            label="Rolling limit cycle"
        )

        plt.scatter(
            limit_cycle[0, 0],
            limit_cycle[1, 0],
            s=60,
            label="Post-impact fixed point"
        )

        plt.xlabel(r"$\theta$ [rad]")
        plt.ylabel(r"$\dot{\theta}$ [rad/s]")
        plt.title("Rimless Wheel Rolling Limit Cycle")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            "assignment_1_graphs/rolling_limit_cycle_2.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
 
    # RETURN MAP
    omega_values, next_omega = calculate_return_map(
        params,
        timestep,
        sim_time
    )

    plt.figure()

    plt.plot(
        omega_values,
        next_omega,
        linewidth=2,
        label=r"Return map $P(\omega)$"
    )

    plt.plot(
        omega_values,
        omega_values,
        linestyle="--",
        label=r"Identity $\omega_{n+1}=\omega_n$"
    )

    plt.scatter(
        omega_fixed,
        omega_fixed,
        s=70,
        label="Fixed point"
    )

    plt.xlabel(r"$\omega_n^+$ [rad/s]")
    plt.ylabel(r"$\omega_{n+1}^+$ [rad/s]")
    plt.title("Rimless Wheel Step-to-Step Return Map")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "assignment_1_graphs/rimless_step_to_step_return_2.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # FLOQUET MULTIPLIER
    floquet = calculate_floquet_multiplier(
        params,
        timestep,
        sim_time
    )

    print("\nFLOQUET MULTIPLIER")
    print(f"Fixed point = {omega_fixed:.6f} rad/s")
    print(f"Floquet multiplier = {floquet:.6f}")
    print(f"Absolute multiplier = {abs(floquet):.6f}")

    if abs(floquet) < 1:
        print("The rolling limit cycle is locally stable.")
    else:
        print("The rolling limit cycle is locally unstable.")

    # CONVERGENCE
    convergence = calculate_convergence(
        params,
        timestep,
        sim_time,
        omega_fixed + 0.20,
        num_contacts=10
    )

    if len(convergence) > 0:

        steps = np.arange(
            1,
            len(convergence) + 1
        )

        plt.figure()

        plt.plot(
            steps,
            convergence,
            marker="o",
            label="Post-impact angular velocity"
        )

        plt.axhline(
            omega_fixed,
            linestyle="--",
            label="Fixed point"
        )

        plt.xlabel("Step number")
        plt.ylabel(r"$\omega_n^+$ [rad/s]")
        plt.title("Convergence to Rolling Fixed Point")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            "assignment_1_graphs/convergence_to_rolling_fixed_2.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

    # REGION OF ATTRACTION
    print("\nREGION OF ATTRACTION")

    theta_values = np.linspace(
        gamma - alpha,
        gamma + alpha,
        50
    )

    omega_values_roa = np.linspace(
        0,
        max(3.0, 1.5 * omega_fixed),
        50
    )

    roa = calculate_roa(
        params,
        timestep,
        sim_time,
        theta_values,
        omega_values_roa
    )

    plot_roa(
        theta_values,
        omega_values_roa,
        roa,
        params,
        limit_cycle
    )

    roa_fraction = np.mean(roa == 1)

    print(
        f"Rolling limit-cycle RoA fraction = "
        f"{roa_fraction:.4f}"
    )

    # INCLINATION SWEEP
    gamma_values = np.radians(
        np.arange(5, 46, 5)
    )

    inclination_roa, inclination_floquet = run_sweep(
        params,
        gamma_values,
        "gamma",
        timestep,
        sim_time
    )

    plot_sweep(
        np.degrees(gamma_values),
        inclination_roa,
        inclination_floquet,
        r"Inclination $\gamma$ [degrees]",
        "assignment_1_graphs/RoA_vs_Inclination_2.png",
        "assignment_1_graphs/floquet_mult_vs_inclination_2.png"
    )

    # SPOKE-NUMBER SWEEP
    spoke_values = np.arange(6, 13)

    spoke_roa, spoke_floquet = run_sweep(
        params,
        spoke_values,
        "N",
        timestep,
        sim_time
    )

    plot_sweep(
        spoke_values,
        spoke_roa,
        spoke_floquet,
        "Number of spokes",
        "assignment_1_graphs/roa_vs_num_spokes_2.png",
        "assignment_1_graphs/floquet_vs_num_spokes_2.png"
    )

    # PRINT SWEEP RESULTS 
    print("\nINCLINATION SWEEP RESULTS")

    for gamma, roa, floquet in zip(
        np.degrees(gamma_values),
        inclination_roa,
        inclination_floquet
    ):

        print(
            f"gamma = {gamma:.1f} deg, "
            f"RoA = {roa:.4f}, "
            f"|lambda| = {abs(floquet):.6f}"
        )

    print("\nSPOKE-NUMBER SWEEP RESULTS")

    for N, roa, floquet in zip(
        spoke_values,
        spoke_roa,
        spoke_floquet
    ):

        print(
            f"N = {N}, "
            f"RoA = {roa:.4f}, "
            f"|lambda| = {abs(floquet):.6f}"
        )
