import numpy as np
def explicit_euler(
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
        raise ValueError("TIMESTEP_MODIFIER must be between 0 and 1")

    x0 = np.asarray(x0, dtype=float)

    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep

    state_traj = np.zeros((len(x0), n_timesteps))
    state_traj[:, 0] = x0

    # events[k] indicates that an event occurred while advancing
    # from time_traj[k - 1] to time_traj[k].
    events = np.zeros(n_timesteps, dtype=bool)

    # Choose an integer number of refined steps so that they exactly
    # cover one ordinary timestep.
    n_refined_steps = int(np.ceil(1.0 / TIMESTEP_MODIFIER))
    refined_timestep = timestep / n_refined_steps

    for step, current_time in enumerate(time_traj[:-1]):
        previous_state = state_traj[:, step].copy()

        # First attempt one normal-sized Euler step.
        candidate_state = (
            previous_state
            + timestep
            * dynamics(current_time, previous_state, params)
        )

        if not event_guard(previous_state, candidate_state, params):
            # No collision occurred during the ordinary step.
            state_traj[:, step + 1] = candidate_state
            continue

        # The ordinary step crossed an event. Discard it and backtrack
        # to the state at the beginning of the interval.
        refined_state = previous_state.copy()
        refined_time = current_time

        for _ in range(n_refined_steps):
            refined_candidate = (
                refined_state
                + refined_timestep
                * dynamics(refined_time, refined_state, params)
            )

            if event_guard(refined_state, refined_candidate, params):
                # Apply the instantaneous collision/reset map.
                refined_candidate = event_dynamics(
                    refined_candidate,
                    params,
                )
                events[step + 1] = True

            refined_state = refined_candidate
            refined_time += refined_timestep

        state_traj[:, step + 1] = refined_state

    return time_traj, state_traj