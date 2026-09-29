import numpy as np


def _rk4_step(time, state, timestep, dynamics, params):
    """Perform one RK4 integration step."""
    k1 = dynamics(time, state, params)
    k2 = dynamics(time + timestep / 2,state + k1 * timestep / 2,params,)
    k3 = dynamics(time + timestep / 2,state + k2 * timestep / 2,params,)
    k4 = dynamics(time + timestep,state + k3 * timestep,params,)
    return state + (timestep / 6) * (k1 + 2 * k2 + 2 * k3 + k4) 


def rk4(
    timestep,
    sim_time,
    x0,
    dynamics,
    params,
    event_guard=lambda previous_state, next_state, params: False,
    event_dynamics=lambda state, params: state,
    TIMESTEP_MODIFIER=1e-2,
):
    if timestep <= 0:
        raise ValueError("timestep must be positive")

    if not 0 < TIMESTEP_MODIFIER < 1:
        raise ValueError(
            "TIMESTEP_MODIFIER must be between 0 and 1"
        )

    x0 = np.asarray(x0, dtype=float)

    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep

    state_traj = np.zeros((len(x0), n_timesteps))
    state_traj[:, 0] = x0

    # events[k] indicates that an event occurred while advancing
    # from time_traj[k - 1] to time_traj[k].
    events = np.zeros(n_timesteps, dtype=bool)

    # Use an integer number of refined steps that exactly covers one
    # ordinary timestep.
    n_refined_steps = int(np.ceil(1.0 / TIMESTEP_MODIFIER))
    refined_timestep = timestep / n_refined_steps

    for step, current_time in enumerate(time_traj[:-1]):
        previous_state = state_traj[:, step].copy()

        candidate_state = _rk4_step(current_time, previous_state, timestep, dynamics, params)

        if not event_guard(previous_state, candidate_state, params):
            # No event: accept the normal-sized step.
            state_traj[:, step + 1] = candidate_state
            continue

        # An event was crossed. Discard the normal step and backtrack.
        refined_state = previous_state.copy()

        for substep in range(n_refined_steps):
            refined_time = current_time + substep * refined_timestep
            refined_candidate = _rk4_step(
                refined_time, refined_state, refined_timestep, dynamics, params
            )

            if event_guard(refined_state, refined_candidate, params):
                # Apply the instantaneous collision/reset map.
                refined_candidate = event_dynamics(refined_candidate, params)
                events[step + 1] = True

            refined_state = refined_candidate

        state_traj[:, step + 1] = refined_state

    return events, time_traj, state_traj
