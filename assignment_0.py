import timeit

import numpy as np
import matplotlib.pyplot as plt

#from models import pendulum as model
from models import bouncing_ball as model
from integrators import rk4 as integrator

# Basic simulation of the pendulum

params = {
    "gravity": 9.81,  # gravity m/s^2)
    "length": 1,  # rod length (m)
    "mass": 0.2,  # point mass at end of rod (kg)
    "damping_coeff": 0.0,  # damping coefficient (kg*m^2/s)
}

initial_state = np.array([np.pi / 4, 0.0])
sim_time = 2


def run_sim(integrator, timestep):
    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state
    for step, t in enumerate(time_traj[:-1]):
        state_traj[:, step+1] = integrator(t, state_traj[:, step], params, timestep, model)
        if state_traj[0, step + 1] < 0:
            state_traj[:, step + 1] = -state_traj[:, step + 1]

    return state_traj


def find_largest_stable_dt(integrator, n_max_sweep=500, tol=1e-2):
    timestep = 1e-5
    largest_stable_dt = None
    for ss in range(n_max_sweep):
        timestep = timestep * 1.2 

        state_traj = run_sim(integrator, timestep)

        kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)
        Etot = kinetic_energy + potential_energy
        Etot0 = Etot[0]

        if not np.all(np.isfinite(Etot)):
            print(f"TimeStep: {timestep:.6g} diverged (non-finite energy).")
            break

        max_rel_dev = np.max(np.abs(Etot - Etot0)) / np.abs(Etot0)
        if max_rel_dev > tol:
            print(f"TimeStep: {timestep:.6g}, Max relative energy deviation: {max_rel_dev:.4g} -> unstable")
            break

        largest_stable_dt = timestep

    return largest_stable_dt


#dt_euler = find_largest_stable_dt(explicit_euler)
#dt_rk4 = find_largest_stable_dt(rk4)
#print(f"Largest stable timestep (Euler): {dt_euler}")
#print(f"Largest stable timestep (RK4):   {dt_rk4}")

## timing comparison 1: same timestep for both integrators
#n_runs = 10
#shared_dt = min(dt_euler, dt_rk4)
#t_euler = timeit.timeit(lambda: run_sim(explicit_euler, shared_dt), number=n_runs)
#t_rk4 = timeit.timeit(lambda: run_sim(rk4, shared_dt), number=n_runs)
#print(f"\nSame dt={shared_dt:.6g}: Euler {t_euler:.4f}s, RK4 {t_rk4:.4f}s ({n_runs} runs)")

## timing comparison 2: each integrator's own largest stable timestep
#t_euler_own = timeit.timeit(lambda: run_sim(explicit_euler, dt_euler), number=n_runs)
#t_rk4_own = timeit.timeit(lambda: run_sim(rk4, dt_rk4), number=n_runs)
#print(
#    f"Own largest-stable dt: Euler dt={dt_euler:.6g} took {t_euler_own:.4f}s, "
#    f"RK4 dt={dt_rk4:.6g} took {t_rk4_own:.4f}s ({n_runs} runs)"
#)

timestep = 1e-5
n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = run_sim(integrator, timestep)
kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)


plt.figure()
plt.plot(time_traj, potential_energy, label="Potential energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Pendulum energy")
plt.legend()
plt.tight_layout()
plt.show()

# TODO: make a phase portrait plot
