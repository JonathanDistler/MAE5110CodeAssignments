import numpy as np
import matplotlib.pyplot as plt

""""
prediction: before carrying out the dynamical derivation, I expect the total energy to be constant
I think that the potential energy and kinetic energy will be sinusoids like the last example
"""

from ball_model import dynamics 
from ball_model import calculate_energy
from explicit_euler import integrate as rk4

# Basic simulation of the pendulum

params = {
    "gravity": 9.81,  # gravity m/s^2)
    "mass": 0.2,  # point mass of the ball
    "damping_coeff": 0.0,  # damping coefficient (kg*m^2/s)
    "k": 1000, #spring constant (N/m)
}


# some set-up
initial_state = np.array([1 , 0.0]) #initial position of 1 m height, 0 velocity

#changing these variables 
timestep = 2e-6 #2e-6 is what I found after sweeping with the code commented out below for the euler
sim_time = 5.0



flag=True
while (timestep<2 and flag):
    #just used the same name as the initial euler file I had made, in the future I would change the python file name from explicit_euler to something else
    time_traj,state_traj=rk4(dynamics, initial_state,timestep,sim_time,params)

    #calculates energy with a model library? probably where negative KE comes from (switched the order to fix)
    kinetic_energy, potential_energy = calculate_energy(state_traj, params)

    a=kinetic_energy[0] + potential_energy[0]
    b=kinetic_energy[-1] + potential_energy[-1] #had originally compared to the middle, this is mroe generalizable 
    if (np.isclose(a,b, rtol=1e-05, atol=1e-06, equal_nan=False)):
        print(f"Timestep is {timestep}")
        timestep+=1e-5
    else:
        print("Total energy is conserved")
        flag=False
        print(f"The best timestep is {timestep}")



#calculates energy with a model library? probably where negative KE comes from (switched the order to fix)
#kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)

plt.figure()
plt.plot(time_traj, potential_energy, label="Potential energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Ball energy")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(state_traj[0, :], state_traj[1, :])
plt.xlabel("Position (m) - x")
plt.ylabel("Velocity (m/s) - f(x)")
plt.title("Phase portrait of the ball")
plt.tight_layout()
plt.show()

