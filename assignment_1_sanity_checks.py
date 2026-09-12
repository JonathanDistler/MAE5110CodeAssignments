
import numpy as np
import matplotlib.pyplot as plt
from models import rimless_wheel as model
from wheel_integrator import rk4 as integrate

# Basic simulation of a rimless wheel
# To check the simulation is running as expected, I can't rely on momentum over the whole simulation, or energy
#So, I propose to graph the theta as a function of time, theta-dot as a function of time, and print-out when the is_touching occurs

params = {
    "gravity": 9.81,  # gravity m/s^2)
    "length": 1,  # rod length (m)
    "mass": 1,  # mass in the middle of the spokes
    "damping_coeff": 0,  # damping coefficient (kg*m^2/s)
    "N": 6, #number of spokes
    "gamma": np.pi/6 #angle of incline
}

sim_time=25 #[s]
num_contacts=50

#set-up
initial_state = np.array([np.pi / 4, 0.0]) #theta, theta-dot

timestep = .001 #when the average change between theta-dot (t-) and theta-dot (t+) begins to diverge, had a separate script for this

time_traj,state_traj, theta_dot_list,_, _=integrate(model.dynamics, model.is_touching, model.reset_params, initial_state,timestep,sim_time,params, num_contacts=num_contacts)

#Time vs theta, Time vs theta-dot graphs
#All graphs show a roughly piece-wise function indicating the change of coordinates with each step 
plt.figure()
plt.plot(time_traj, state_traj[0, :], label="Angle vs. Time")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title("Time vs. Angle (rad)")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(time_traj, state_traj[1, :], label="Angular Velocity vs. Time")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (rad/s)")
plt.title("Time vs. Angular Velocity")
plt.legend()
plt.tight_layout()
plt.show()


#Return map plotting
v_k=theta_dot_list[:-1]
v_next=theta_dot_list[1:]
plt.figure()
plt.plot(v_k,v_next,label="Return Map")
v_min=min(v_k)
v_max=max(v_k)

#such that there is a minimum number of contacts detected to get arithmetic mean
num_contacts_detected=min(5,len(theta_dot_list))
v_fixed=np.mean(theta_dot_list[-num_contacts_detected:])
print(f"The estimated fixed point is: {v_fixed} rad/s")

#Return map plotting
plt.plot([v_min,v_max],[v_min,v_max],label="y=x")
plt.xlabel(r"$\dot{\theta}_{k,t^+}$ (rad/s)")
plt.ylabel(r"$\dot{\theta}_{k+1,t^+}$ (rad/s)")
plt.title("Rimless Wheel Return Map")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#making sure that the contact number is roughly the same over the last few points, which is the case
plt.figure()
plt.plot(range(1, len(theta_dot_list) + 1), theta_dot_list)
plt.xlabel("Contact number")
plt.ylabel(r"Post-impact $\dot{\theta}_{t^+}$ (rad/s)")
plt.title("Post-Impact Velocity")
plt.grid()
plt.tight_layout()
plt.show()


#Floquet multiplier

#helper function 
def calculate_return_map(post_impact_velocity,timestep,params,sim_time):
    N=params["N"]
    two_alpha=(2*np.pi)/N
    alpha=two_alpha/2
    theta_plus=params["gamma"]-alpha

    initial_state = np.array([theta_plus, post_impact_velocity]) #theta, theta-dot
    _,_,theta_dot_list=integrate(model.dynamics,model.is_touching,model.reset_params,initial_state,timestep,sim_time,params,num_contacts=1)

    #if there are no contacts
    if len(theta_dot_list)==0:
        print("No contact happened during the simulation")
        return(None)
    return(theta_dot_list[0])

#list of perturbation sizes, will use this to figure out when it diverges
delta_arr=np.array([ 0.2, .15, 0.1, .075, 0.05, 0.025])
floquet_vals=[]
for delta in delta_arr:
    p_minus=calculate_return_map(v_fixed-delta,timestep,params,sim_time)
    p_plus=calculate_return_map(v_fixed+delta,timestep,params,sim_time)
    floquet=(p_plus-p_minus)/(2*delta)
    floquet_vals.append(floquet)
    print(f"The associated Floquet Value is {floquet} for a perturbation of {delta}")

#plots perturbation size versus eigenvalue of 1 (upper-bound of stability for discrete time)
plt.figure()
plt.plot(delta_arr, np.abs(floquet_vals), label="Estimated Floquet Multiplier", color='red')
plt.axhline(1, label="||Lambda||=1", color='blue')
plt.xlabel("Perturbation size (rad/s)")
plt.ylabel("Eigenvalue")
plt.title("Floquet Multiplier Convergence")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


#Gamma Sweep

#gammas being iterated over
gamma_vals = np.linspace(0, np.pi / 2, 50)

#floquet multipliers for gamma sweep
floquet_gammas = []

#fixed points for gammas sweep
fixed_point_gammas = []

# Use one perturbation size for the gamma sweep, based on the previous tests .05 is good
delta_gamma = 0.05

for gamma in gamma_vals:

    # Copy the original parameters so that the original params dictionary is not changed
    sweep_params = params.copy()
    sweep_params["gamma"] = gamma

    # Simulate the wheel until it reaches the desired number of contacts
    _, _, theta_dot_gamma, _, _ = integrate(model.dynamics, model.is_touching, model.reset_params, initial_state, timestep, sim_time, sweep_params, num_contacts=num_contacts)

    # Make sure there are enough contacts occurred to estimate the steady-state fixed point
    if len(theta_dot_gamma) < 5:
        print(f"Not enough contacts occurred for {np.degrees(gamma)} degrees")
        #to match matrix sizes
        floquet_gammas.append(np.nan)
        fixed_point_gammas.append(np.nan)
    else:
        # Estimate the fixed point using the last 5 contacts
        num_contacts_detected = min(5, len(theta_dot_gamma))

        v_fixed_gamma = np.mean(theta_dot_gamma[-num_contacts_detected:])
        fixed_point_gammas.append(v_fixed_gamma)

        # Calculate the return map on either side of the fixed point
        p_minus = calculate_return_map(v_fixed_gamma - delta_gamma, timestep, sweep_params, sim_time)

        p_plus = calculate_return_map(v_fixed_gamma + delta_gamma,timestep, sweep_params, sim_time)

        # Checks whether both return-map calculations succeeded 
        if p_minus is None or p_plus is None:
            floquet_gammas.append(np.nan)
            print(f"Floquet calculation failed for {np.degrees(gamma)} degrees")
        else:

            # Central-difference estimate of Floquet multiplier for the local slope
            floquet_val = (p_plus - p_minus) / (2 * delta_gamma)
            floquet_gammas.append(floquet_val)
            print(f"gamma={np.degrees(gamma)} degrees; fixed point={v_fixed_gamma} rad/s; Floquet multiplier={floquet_val}")


# Plot Floquet Multiplier vs. Incline
plt.figure()
plt.plot( np.degrees(gamma_vals), np.abs(floquet_gammas),'o-', label="Estimated Floquet Multiplier",color='red')
#eigenvalue of 1 reference (horizontal line)
plt.axhline(1,linestyle="--", label=r"||lambda||=1", color='blue')
plt.xlabel(r"Incline $\gamma$ (degrees)")
plt.ylabel("||lambda||")
plt.title("Floquet Multiplier vs. Slope Incline")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plot Fixed Point vs. Incline
plt.figure()
plt.plot(np.degrees(gamma_vals), fixed_point_gammas, 'o-', color='red',label=r"Fixed Point Theta-Dot (rad/s)")
plt.xlabel(r"Incline $\gamma$ (degrees)")
plt.ylabel(r"Post-Impact Fixed Point (rad/s)")
plt.title("Fixed Point vs. Slope Incline")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


#Leg Number Sweep

#leg numbers being iterated over
leg_vals = np.arange(6,13)

#floquet multipliers for leg sweep
floquet_legs = []

#fixed points for gammas sweep
fixed_point_legs = []

# Use one perturbation size for the leg sweep, based on the previous tests .05 is good
delta_leg = 0.05

for legs in leg_vals:

    # Copy the original parameters so that the original params dictionary is not changed
    sweep_params = params.copy()
    sweep_params["N"] = legs

    # Simulate the wheel until it reaches the desired number of contacts
    _, _, theta_dot_legs = integrate(model.dynamics, model.is_touching, model.reset_params, initial_state, timestep, sim_time, sweep_params, num_contacts=num_contacts)

    # Make sure there are enough contacts occurred to estimate the steady-state fixed point
    if len(theta_dot_legs) < 5:
        print(f"Not enough contacts occurred for {legs} legs")
        #to match matrix sizes
        floquet_legs.append(np.nan)
        fixed_point_legs.append(np.nan)
    else:
        # Estimate the fixed point using the last 5 contacts
        num_contacts_detected = min(5, len(theta_dot_legs))

        v_fixed_legs = np.mean(theta_dot_legs[-num_contacts_detected:])
        fixed_point_legs.append(v_fixed_legs)

        # Calculate the return map on either side of the fixed point
        p_minus = calculate_return_map(v_fixed_legs - delta_leg, timestep, sweep_params, sim_time)

        p_plus = calculate_return_map(v_fixed_legs + delta_leg,timestep, sweep_params, sim_time)

        # Checks whether both return-map calculations succeeded 
        if p_minus is None or p_plus is None:
            floquet_legs.append(np.nan)
            print(f"Floquet calculation failed for {legs} legs")
        else:

            # Central-difference estimate of Floquet multiplier for the local slope
            floquet_val = (p_plus - p_minus) / (2 * delta_leg)
            floquet_legs.append(floquet_val)
            print(f"Leg Number={legs} legs; fixed point={v_fixed_legs} rad/s; Floquet multiplier={floquet_val}")


# Plot Floquet Legs vs. Incline
plt.figure()
plt.plot(leg_vals, np.abs(floquet_legs),'o-', label="Estimated Floquet Multiplier",color='red')
#eigenvalue of 1 reference (horizontal line)
plt.axhline(1,linestyle="--", label=r"||lambda||=1", color='blue')
plt.xlabel("Number of Legs")
plt.ylabel("||lambda||")
plt.title("Floquet Multiplier vs. Number of Legs")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plot Fixed Point vs. Number of Legs
plt.figure()
plt.plot(leg_vals, fixed_point_legs, 'o-', color='red',label=r"Fixed Point Theta-Dot (rad/s)")
plt.xlabel("Number of Legs")
plt.ylabel(r"Post-Impact Fixed Point (rad/s)")
plt.title("Fixed Point vs. Number of Legs")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
