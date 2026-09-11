import numpy as np
from scipy.integrate import solve_ivp



def explicit_euler(t, state0, params, timestep, model):
    state1 = state0 + timestep * model.dynamics(t, state0, params)
    return state1


def rk4(t, state0, params, timestep, model):
    state = state0
    k1 = model.dynamics(t, state, params)
    k2 = model.dynamics(t + timestep / 2, state + timestep / 2 * k1, params)
    k3 = model.dynamics(t + timestep / 2, state + timestep / 2 * k2, params)
    k4 = model.dynamics(t + timestep, state + timestep * k3, params)
    state1 = state + timestep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return state1

#good integrator, matches convention in rimless wheel model of name - closest to a reasonable integrator
#could probably use this over IVP with guard or delete this and keep IVP with guard
def rk4_with_guard(t, state, params, timestep, model):
    k1 = model.dynamics(t, state, params)
    k2 = model.dynamics(t + timestep / 2, state + timestep / 2 * k1, params)
    k3 = model.dynamics(t + timestep / 2, state + timestep / 2 * k2, params)
    k4 = model.dynamics(t + timestep, state + timestep * k3, params)
    #reasonably good name
    state_next = state + timestep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    #vague name, what does "g" represent? 
    g_current = model.guard(state, params)
    g_next = model.guard(state_next, params)

    #this makes sense given the float output of the guard function, boolean is typically easier to follow
    if g_current < 0 and g_next >= 0:
        frac = g_current / (g_current - g_next)
        state_impact = state + frac * (state_next - state)
        state_next = model.reset(state_impact, params)

    #found it helpful to return the time traj, state traj, theta_dot_pos, number of contacts, pre impact state, and post impact states
    return state_next

#how were the tolerances decided? could be useful to include in sanity checks 
def ivp_with_guard(initial_state, params, model, sim_time,
                       max_collisions=10, rtol=1e-4, atol=1e-5):
    def guard_event(t, state, params):
        return model.guard(state, params)

    guard_event.terminal = True
    guard_event.direction = 1  # only trigger when guard crosses zero going positive - what about a negative initial ang-velocity that rolls backwards briefly before forwards? 

    t = 0.0
    state = np.array(initial_state, dtype=float)

    #good inclusion of time trajectory, state trajectory, collision times, and collision states
    time_traj = [t]
    state_traj = [state.copy()]
    collision_times = []
    collision_states = []

    #I found it useful to use max collisions or sim-time, but not both
    for _ in range(max_collisions):
        if t >= sim_time:
            break

        sol = solve_ivp(
            model.dynamics,
            (t, sim_time),
            state,
            args=(params,),
            events=guard_event,
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )

        # append all solver-chosen points except the first (already stored)
        time_traj.extend(sol.t[1:])
        state_traj.extend(sol.y[:, 1:].T)

        #maybe a better name instead of "sol"
        if sol.status == 1:  # guard event triggered - as a note, I find it easier to review with Booleans rather than ints
            t_impact = sol.t_events[0][0]
            state_impact = sol.y_events[0][0]

            state = model.reset(state_impact, params)
            t = t_impact

            #good idea to use copy so state isn't overwritten - could possibly define globally?
            collision_times.append(t_impact)
            collision_states.append(state.copy())

            # record the post-reset state as the next trajectory point too
            time_traj.append(t)
            state_traj.append(state.copy())
        else:
            # reached sim_time or stalled with no more crossings
            # I think the logic needs to be more fleshed out here, as of now it seems like it simulations
            #a bunch of collisions, until the time is reached, but if the guard is never triggered, it will just break out of the loop and return the current state
            break

    time_traj = np.array(time_traj)
    state_traj = np.array(state_traj).T  # shape (2, N)

    #good return statement! 
    return time_traj, state_traj, collision_times, collision_states