import numpy as np
import pytest
from integrators import rk4
from models import pendulum as model


def test_energy_conservation():
    params = model.generate_params()
    # Energy is conserved only without damping or external torque.
    params["damping_coeff"] = 0.0
    params["torque"] = 0.0


    timestep = 0.01
    time_traj = np.arange(100) * timestep
    state_traj = np.zeros((2, len(time_traj)))
    # Start away from equilibrium so the pendulum actually moves
    state_traj[:, 0] = [np.pi / 4, 0.0]


    for step, time in enumerate(time_traj[:-1]):
        state_traj[:, step + 1] = rk4(model.dynamics, time, state_traj[:, step], timestep, params)


    kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)
    total_energy = kinetic_energy + potential_energy


    # Checks to see if they are close, given numerical error
    assert total_energy==pytest.approx(total_energy[0]) #could also use np.isclose, but using native approx feature for conciseness


def test_energy_loss():
    params=model.generate_params()
    params["damping_coeff"]=1.0
    params["torque"]=0.0


    timestep=0.01
    time_traj=np.arange(100)*timestep #carry out long enough that the pendulum should lose all its energy
    state_traj=np.zeros((2,len(time_traj)))
    # Start away from the equilibirum so the pendulum actually moves
    state_traj[:,0]=[np.pi/4, 0.0]


    for step, time in enumerate(time_traj[:-1]):
        state_traj[:, step+1]=rk4(model.dynamics, time, state_traj[:, step], timestep, params)
    kinetic_energy, potential_energy=model.calculate_energy(state_traj, params)
    total_energy=kinetic_energy + potential_energy
    difference=(total_energy[-1]-total_energy[0])
    assert (not(total_energy[-1]==pytest.approx(total_energy[0])) and difference<0) #tests if the energy is lost, in this case a sizable amount

def test_torque():
    params=model.generate_params()
    params["damping_coeff"]=0.0
    params["torque"]=10.0 #huge torque, should increase KE by a lot


    timestep=0.01
    time_traj=np.arange(100)*timestep #carry out long enough that the pendulum should gain a lot of energy
    state_traj=np.zeros((2,len(time_traj)))
    # Start away from the equilibirum so the pendulum actually moves
    state_traj[:,0]=[np.pi/4, 0.0]


    for step, time in enumerate(time_traj[:-1]):
        state_traj[:, step+1]=rk4(model.dynamics, time, state_traj[:, step], timestep, params)
    kinetic_energy, potential_energy=model.calculate_energy(state_traj, params)
    total_energy=kinetic_energy + potential_energy
    difference=(total_energy[-1]-total_energy[0])
    assert (not(total_energy[-1]==pytest.approx(total_energy[0])) and difference>1e-3) #tests if energy is added - which with a torque, this increase angular velocity a lot

