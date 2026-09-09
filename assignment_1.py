import numpy as np
import matplotlib.pyplot as plt
from models import rimless_wheel as model
from wheel_integrator import rk4 as integrate

# BASIC SIMULATION OF A RIMLESS WHEEL
# A lot of the work is related to: https://underactuated.csail.mit.edu/simple_legs.html
# Forward walking only occurs if: w1=sqrt(2*(g/L)(1-cos(gamma-alpha)))
# There is a loss of velocity at each impact proportional to: cos(2alpha)

# PARAMETERS
params = {
    "gravity": 9.81,
    "length": 1,
    "mass": 1,
    "damping_coeff": 0,
    "N": 6,
    "gamma": np.pi / 6
}

#will sim until time is reached or number of contacts is reached
sim_time = 20
num_contacts = 50
initial_state = np.array([np.pi / 4, 0.0])
timestep = 0.001 #determined empirically from theta-dot(plus)-theta-dot(minus) for a step to see when it diverges

# GEOMETRY
N = params["N"]
gamma = params["gamma"]

#derivation of alpha 
two_alpha = (2 * np.pi) / N
alpha = two_alpha / 2

# FIXED POINT
def calculate_fixed_point(params):

    gravity = params["gravity"]
    length = params["length"]
    N = params["N"]
    gamma = params["gamma"]

    two_alpha = (2 * np.pi) / N
    alpha = two_alpha / 2

    impact_factor = np.cos(two_alpha)

    # In the stance phase, mechanical energy is conserved, the decrease in potential energy becomes kinetic energy
    # After contact, there is an energy loss due to impact, this quantity is the increase in angular-velocity-squared in the pendulum swing
    velocity_squared_gain = (4 * gravity / length* np.sin(alpha)* np.sin(gamma))

    if gamma <= 0:
        return np.nan

    # Same as the rolling fixed point, defined in the MIT Underactuated Robotics material
    omega_fixed = np.sqrt(impact_factor**2* velocity_squared_gain/ (1 - impact_factor**2))

    return omega_fixed

# FORWARD WALKING THRESHOLD
def calculate_walking_threshold(params):

    gravity = params["gravity"]
    length = params["length"]
    N = params["N"]
    gamma = params["gamma"]

    alpha = np.pi / N
    # threshold to determine if the walker will actually walk, if it falls below, then can just jump through time
    threshold_squared = (2* gravity/ length* (1- np.cos(gamma - alpha)))

    if threshold_squared <= 0:
        return 0.0

    return np.sqrt(threshold_squared)

# LIMIT CYCLE
def calculate_limit_cycle(params,timestep, sim_time):

    N = params["N"]
    gamma = params["gamma"]

    two_alpha = (2 * np.pi) / N
    alpha = two_alpha / 2

    omega_fixed = calculate_fixed_point(params)

    if not np.isfinite(omega_fixed):
        return None, None, np.nan

    # Start immediately after impact, the post-impact leg angle is gamma-alpha 
    initial_state = np.array([gamma - alpha, omega_fixed])

    # Simulate until the next impact
    (time_result, state_result, theta_dot_pos, contact_indices, pre_impact_states, post_impact_states) = integrate(model.dynamics, model.is_touching, model.reset_params, initial_state, timestep, sim_time, params, num_contacts=1)

    theta_tolerance=.02
    omega_tolerance=.02
    #checks to see if no contacts over span and within the tolerance
    standing = (len(contact_indices) == 0 and np.max(np.abs(state_result[0])) < theta_tolerance and np.max(np.abs(state_result[1])) < omega_tolerance)

    if standing:
        return 2

    contact_index = contact_indices[0]

    # Keep the continuous stance trajectory before impact
    limit_cycle = state_result[:, :contact_index]
    time_cycle = time_result[:contact_index]

    return (time_cycle, limit_cycle, omega_fixed)


# RETURN MAP
def return_map(omega_initial, params, timestep, sim_time):

    N = params["N"]
    gamma = params["gamma"]

    alpha = np.pi / N

    # Start immediately after the impact provides a new state to create return map 
    initial_state = np.array([gamma - alpha, omega_initial])

    (time_result, state_result,theta_dot_pos, contact_indices, pre_impact_states, post_impact_states) = integrate(model.dynamics, model.is_touching, model.reset_params, initial_state, timestep, sim_time, params, num_contacts=1)

    if len(theta_dot_pos) == 0:
        return np.nan

    # First stored contact velocity is the next post-impact velocity
    return theta_dot_pos[0]

# RETURN MAP DATA
def calculate_return_map(params, timestep, sim_time, omega_min=0.1, omega_max=3.5, num_points=100):

    omega_values = np.linspace(omega_min, omega_max, num_points)

    next_omega = np.zeros(num_points)

    for i, omega in enumerate(omega_values):

        next_omega[i] = return_map(omega, params, timestep, sim_time)

    valid = np.isfinite(next_omega)

    return (omega_values[valid],next_omega[valid])

# FLOQUET MULTIPLIER (eigenvalue for discrete system)
def calculate_floquet_multiplier(params, timestep, sim_time, delta=0.05):

    omega_fixed = calculate_fixed_point(params)

    if not np.isfinite(omega_fixed):
        return np.nan

    omega_plus = return_map(omega_fixed + delta, params, timestep, sim_time)

    omega_minus = return_map(omega_fixed - delta,params, timestep, sim_time)

    if not (np.isfinite(omega_plus) and np.isfinite(omega_minus)):
        return np.nan

    floquet_multiplier = (omega_plus - omega_minus) / (2 * delta)
    return floquet_multiplier

# CONVERGENCE TEST
def calculate_convergence(params, timestep, sim_time, initial_omega, num_contacts=10):

    N = params["N"]
    gamma = params["gamma"]
    alpha = np.pi / N
    initial_state = np.array([gamma - alpha, initial_omega])
    (time_result, state_result, theta_dot_pos, contact_indices, pre_impact_states, post_impact_states) = integrate(model.dynamics, model.is_touching, model.reset_params, initial_state, timestep, sim_time, params, num_contacts=num_contacts)

    return np.array(theta_dot_pos)


# ATTRACTOR CLASSIFICATION
def classify_initial_condition(initial_state, params, timestep, sim_time, omega_tolerance=0.02):

    (time_result, state_result, theta_dot_pos, contact_indices, pre_impact_states, post_impact_states) = integrate(model.dynamics, model.is_touching, model.reset_params, initial_state, timestep, sim_time, params, num_contacts=None)

    omega_fixed = calculate_fixed_point(params)

    # ROLLING LIMIT CYCLE
    if (len(theta_dot_pos) >= 5 and np.isfinite(omega_fixed)):

        recent_velocities = np.array(theta_dot_pos[-5:])

        if np.all(np.abs(recent_velocities- omega_fixed)< omega_tolerance):
            return 1 #means its rolling

    # NON-WALKING / FALLING
    return 0

# REGION OF ATTRACTION
def calculate_roa(params, timestep, sim_time, theta_values, omega_values):

    roa = np.zeros((len(omega_values), len(theta_values)))

    for i, omega in enumerate(omega_values):

        print(f"RoA row {i + 1}/{len(omega_values)}")

        for j, theta in enumerate(theta_values):
            initial_state = np.array([theta, omega])
            roa[i, j] = classify_initial_condition(initial_state, params, timestep,sim_time)

    return roa

# PLOT REGION OF ATTRACTION
def plot_roa(theta_values,omega_values, roa,params,limit_cycle=None):

    gamma = params["gamma"]

    plt.figure()
    plt.plot(theta_values, omega_values, linewidth=2, label="Rolling limit cycle")
    plt.xlabel(r"$\theta$ [rad]")
    plt.ylabel(r"$\dot{\theta}$ [rad/s]")
    plt.title(f"Region of Attraction, N = {params['N']}, $\\gamma$ = {np.degrees(gamma):.1f}$^\\circ$")
    if limit_cycle is not None:
        plt.plot(limit_cycle[0], limit_cycle[1], linewidth=2.5, label="Rolling limit cycle")
        plt.scatter(limit_cycle[0, 0], limit_cycle[1, 0], s=70, label="Post-impact fixed point")
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("assignment_1_graphs/RoA_N6_g30.png", dpi=300, bbox_inches="tight")
    plt.show()


# INCLINATION SWEEP
def inclination_sweep(params, gamma_values, timestep, sim_time):

    roa_sizes = []
    floquet_values = []

    for gamma in gamma_values:
        print(f"INCLINATION = {np.degrees(gamma):.2f} degrees")
        sweep_params = params.copy()

        sweep_params["gamma"] = gamma

        # FLOQUET MULTIPLIER
        floquet = calculate_floquet_multiplier(sweep_params, timestep, sim_time)

        floquet_values.append(floquet)

        # ROA
        N = sweep_params["N"]

        alpha = np.pi / N

        theta_values = np.linspace(gamma - alpha, gamma + alpha,35)
        omega_fixed = calculate_fixed_point(sweep_params)
        if np.isfinite(omega_fixed):

            omega_max = max(3.0, 1.5 * omega_fixed)

        else:

            omega_max = 3.0

        omega_values = np.linspace(0, omega_max, 35)

        #calculates region of attraction 
        roa = calculate_roa(sweep_params, timestep, sim_time, theta_values, omega_values)
        roa_fraction = np.mean(roa == 1)
        roa_sizes.append(roa_fraction)

    return np.array(roa_sizes), np.array(floquet_values)


# SPOKE-NUMBER SWEEP
def spoke_sweep(params, spoke_values, timestep, sim_time):

    roa_sizes = []
    floquet_values = []

    for N in spoke_values:

        print(f"NUMBER OF SPOKES = {N}")
        sweep_params = params.copy()
        sweep_params["N"] = N

        # FLOQUET MULTIPLIER
        floquet = calculate_floquet_multiplier(sweep_params, timestep, sim_time)

        floquet_values.append(floquet)

        # ROA
        gamma = sweep_params["gamma"]
        alpha = np.pi / N

        theta_values = np.linspace(gamma - alpha, gamma + alpha, 35)

        omega_fixed = calculate_fixed_point(sweep_params)

        if np.isfinite(omega_fixed):

            omega_max = max(3.0, 1.5 * omega_fixed)

        else:

            omega_max = 3.0

        omega_values = np.linspace(0, omega_max, 35)

        roa = calculate_roa(sweep_params,timestep,sim_time,theta_values, omega_values)
        roa_fraction = np.mean(roa == 1)
        roa_sizes.append(roa_fraction)

    return np.array(roa_sizes), np.array(floquet_values)

# PLOT INCLINATION SWEEP
def plot_inclination_sweep(gamma_values, roa_values,floquet_values):
    gamma_degrees = np.degrees(gamma_values)

    # ROA
    plt.figure()
    plt.plot(gamma_degrees, roa_values, marker="o", label="Rolling limit-cycle RoA")
    plt.xlabel(r"Inclination $\gamma$ [degrees]")
    plt.ylabel("Fraction of sampled state space")
    plt.title("RoA vs Inclination")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("assignment_1_graphs/RoA_vs_Inclination.png", dpi=300, bbox_inches="tight")
    plt.show()

    # FLOQUET MULTIPLIER
    plt.figure()
    plt.plot(gamma_degrees, np.abs(floquet_values), marker="o", label=r"$|\lambda|$")
    plt.axhline(1, linestyle="--", label="Stability boundary")
    plt.xlabel(r"Inclination $\gamma$ [degrees]")
    plt.ylabel(r"$|\lambda|$")
    plt.title("Floquet Multiplier vs Inclination")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("assignment_1_graphs/floquet_mult_vs_inclination.png", dpi=300, bbox_inches="tight")
    plt.show()

# PLOT SPOKE SWEEP
def plot_spoke_sweep(spoke_values, roa_values, floquet_values):
    # ROA
    plt.figure()
    plt.plot(spoke_values, roa_values, marker="o", label="Rolling limit-cycle RoA")
    plt.xlabel("Number of spokes")
    plt.ylabel("Fraction of sampled state space")
    plt.title("RoA vs Number of Spokes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("assignment_1_graphs/roa_vs_num_spokes.png", dpi=300, bbox_inches="tight")
    plt.show()

    # FLOQUET MULTIPLIER
    plt.figure()
    plt.plot(spoke_values, np.abs(floquet_values), marker="o", label=r"$|\lambda|$")
    plt.axhline(1, linestyle="--", label="Stability boundary")
    plt.xlabel("Number of spokes")
    plt.ylabel(r"$|\lambda|$")
    plt.title("Floquet Multiplier vs Number of Spokes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("assignment_1_graphs/floquet_vs_num_spokes.png", dpi=300, bbox_inches="tight")
    plt.show()

# MAIN ANALYSIS
if __name__ == "__main__":

    # BASIC GEOMETRY
    N = params["N"]
    gamma = params["gamma"]

    two_alpha = (2 * np.pi) / N
    alpha = two_alpha / 2

    # FIXED POINT
    omega_fixed = calculate_fixed_point(params)
    walking_threshold = calculate_walking_threshold(params)

    # LIMIT CYCLE
    (time_cycle, limit_cycle, omega_fixed) = calculate_limit_cycle(params, timestep, sim_time)

    # PRINT LIMIT CYCLE INFORMATION
    print("LIMIT CYCLE")
    print(f"Fixed point velocity = {omega_fixed:.6f} rad/s")
    print(f"Forward walking threshold = {walking_threshold:.6f} rad/s")
    print(f"Post-impact theta = {gamma - alpha:.6f} rad")
    print(f"Pre-impact theta = {gamma + alpha:.6f} rad")

    if limit_cycle is not None:
        print(f"Number of points = {limit_cycle.shape[1]}")

    # PLOT LIMIT CYCLE
    if limit_cycle is not None:

        plt.figure()
        plt.plot(limit_cycle[0], limit_cycle[1], linewidth=2, label="Rolling limit cycle")
        plt.scatter(limit_cycle[0, 0], limit_cycle[1, 0], s=60, label="Post-impact fixed point")
        plt.xlabel(r"$\theta$ [rad]")
        plt.ylabel(r"$\dot{\theta}$ [rad/s]")
        plt.title("Rimless Wheel Rolling Limit Cycle")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("assignment_1_graphs/rolling_limit_cycle.png", dpi=300, bbox_inches="tight")
        plt.show()

    # RETURN MAP
    omega_values, next_omega = calculate_return_map(params, timestep,sim_time, omega_min=0.1,omega_max=3.5,num_points=100)

    plt.figure()
    plt.plot(omega_values, next_omega, linewidth=2, label=r"Return map $P(\omega)$")
    plt.plot(omega_values, omega_values, linestyle="--", label=r"Identity $\omega_{n+1}=\omega_n$")
    plt.scatter(omega_fixed, omega_fixed, s=70, label="Fixed point")
    plt.xlabel(r"$\omega_n^+$ [rad/s]")
    plt.ylabel(r"$\omega_{n+1}^+$ [rad/s]")
    plt.title("Rimless Wheel Step-to-Step Return Map")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("assignment_1_graphs/rimless_step_to_step_return.png", dpi=300, bbox_inches="tight")
    plt.show()

    # FLOQUET MULTIPLIER
    floquet_multiplier = calculate_floquet_multiplier(params, timestep, sim_time, delta=0.05)

    print("FLOQUET MULTIPLIER")
    print(f"Fixed point = {omega_fixed:.6f} rad/s")
    print(f"Floquet multiplier = {floquet_multiplier:.6f}")
    print(f"Absolute multiplier = {abs(floquet_multiplier):.6f}")

    if abs(floquet_multiplier) < 1:
        print("The rolling limit cycle is locally stable.")

    else:
        print("The rolling limit cycle is locally unstable.")

    # CONVERGENCE
    convergence = calculate_convergence(params, timestep, sim_time, omega_fixed + 0.20, num_contacts=10)

    if len(convergence) > 0:
        contact_numbers = np.arange(1, len(convergence) + 1)

        plt.figure()
        plt.plot(contact_numbers, convergence, marker="o", label="Post-impact angular velocity")
        plt.axhline(omega_fixed, linestyle="--", label="Fixed point")
        plt.xlabel("Step number")
        plt.ylabel(r"$\omega_n^+$ [rad/s]")
        plt.title("Convergence to Rolling Fixed Point")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("assignment_1_graphs/convergence_to_rolling_fixed.png", dpi=300, bbox_inches="tight")
        plt.show()

    # REGION OF ATTRACTION
    theta_values = np.linspace(gamma - alpha, gamma + alpha, 50)
    omega_values_roa = np.linspace(0, max(3.0, 1.5 * omega_fixed), 50)

    print("REGION OF ATTRACTION")
    roa = calculate_roa(params, timestep, sim_time, theta_values, omega_values_roa)

    plot_roa(theta_values, omega_values_roa, roa, params, limit_cycle=limit_cycle)

    roa_fraction = np.mean(roa == 1)

    print(f"Rolling limit-cycle RoA fraction = {roa_fraction:.4f}")

    # INCLINATION SWEEP
    gamma_values = np.radians(np.arange(5, 46, 5))

    (inclination_roa, inclination_floquet) = inclination_sweep(params, gamma_values, timestep, sim_time)

    plot_inclination_sweep(gamma_values, inclination_roa, inclination_floquet)

    # SPOKE-NUMBER SWEEP
    spoke_values = np.arange(6, 13)

    (spoke_roa, spoke_floquet) = spoke_sweep(params, spoke_values, timestep, sim_time)

    plot_spoke_sweep(spoke_values, spoke_roa, spoke_floquet)

    # PRINT SWEEP RESULTS
    print("INCLINATION SWEEP RESULTS")
    for i, gamma_value in enumerate(gamma_values):

        print(
            f"gamma = "
            f"{np.degrees(gamma_value):.1f} deg, "
            f"RoA = "
            f"{inclination_roa[i]:.4f}, "
            f"|lambda| = "
            f"{abs(inclination_floquet[i]):.6f}"
        )

    print("SPOKE-NUMBER SWEEP RESULTS")

    for i, spoke_value in enumerate(spoke_values):

        print( 
            f"N = "
            f"{spoke_value}, "
            f"RoA = "
            f"{spoke_roa[i]:.4f}, "
            f"|lambda| = "
            f"{abs(spoke_floquet[i]):.6f}"
        )