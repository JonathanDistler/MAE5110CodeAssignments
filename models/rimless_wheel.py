import numpy as np

def generate_params():
    params = {
        "gravity": 9.8,  # gravity m/s^2)
        "length": 1,  # rod length (m)
        "mass": 1,  # point mass at end of rod (kg)
        "slope_angle": 0.08,  # slope angle (radians)
        "half_spoke_angle": np.pi/8,  # half spoke angle (radians)
    }

    return params

def dynamics(t, state, params):
    """Continuous-time stance dynamics: theta_ddot = (g/l) sin(theta)."""
    g = params["gravity"]
    l = params["length"]

    theta, theta_dot = state
    theta_ddot = (g / l) * np.sin(theta)

    return np.array([theta_dot, theta_ddot])

def guard(state, params):
    gamma = params["slope_angle"]
    alpha = params["half_spoke_angle"]
    theta = state[0]

    return theta - (gamma + alpha)


def reset(state, params):
    gamma = params["slope_angle"]
    alpha = params["half_spoke_angle"]
    theta_dot = state[1]

    theta_next = gamma - alpha
    theta_dot_next = theta_dot * np.cos(2 * alpha)

    return np.array([theta_next, theta_dot_next])










#def dynamics(t, state_current, params):
#    g = params["gravity"]
#    l = params["length"]
#    m = params["mass"]
#    gamma = params["slope_angle"]
#    alpha = params["half_spoke_angle"]
#    w1 = params["upper_threshold"]
#    w2 = params["lower_threshold"]


#    #angle_previous = state_previous[0]
#    #angular_velocity_previous = state_previous[1]

#    angle_current = state_current[0]
#    angular_velocity_current = state_current[1]

#    # detect collision
#    if angle_current <= gamma - alpha or angle_current >= gamma + alpha:
#        # collision with leading spoke
#        angular_velocity_next = angular_velocity_current*np.cos(2*alpha)
#    else:
#        if gamma <= alpha:
#            if angular_velocity_current > w1:
#                angular_velocity_next = np.cos(2*alpha)*np.sqrt(angular_velocity_current**2 + 4*g/l*(np.sin(alpha)-np.sin(gamma)))
#            elif angular_velocity_current > w2:
#                angular_velocity_next = -angular_velocity_current*np.cos(2*alpha)
#            else:
#                angular_velocity_next =-np.cos(2*alpha)*np.sqrt(angular_velocity_current**2 - 4*g/l*(np.sin(alpha)+np.sin(gamma)))
#        else:
#            if angular_velocity_current >= 0:
#                angular_velocity_next = np.cos(2*alpha)*np.sqrt(angular_velocity_current**2 + 4*g/l*(np.sin(alpha)-np.sin(gamma)))
#            elif angular_velocity_current > w2:
#                angular_velocity_next = -angular_velocity_current*np.cos(2*alpha)
#            else:
#                angular_velocity_next =-np.cos(2*alpha)*np.sqrt(angular_velocity_current**2 - 4*g/l*(np.sin(alpha)+np.sin(gamma)))

#    angular_acceleration_next = -g/l*np.sin(angle_current)


#    state_next = np.array([angular_velocity_next, angular_acceleration_next])
#    return state_next


#def generate_params():
#    g = 9.8
#    l = 1
#    m = 1
#    gamma = 0.08
#    alpha = np.pi/8

#    upper_threshold = np.sqrt(2*g/l*(1-np.cos(gamma-alpha)))
#    lower_threshold = -np.sqrt(2*g/l*(1-np.cos(gamma+alpha)))

#    params = {
#        "gravity": g,  # gravity m/s^2)
#        "length": l,  # rod length (m)
#        "mass": m,  # point mass at end of rod (kg)
#        "slope_angle": gamma,  # slope angle (radians)
#        "half_spoke_angle": alpha,  # half spoke angle (radians)
#        "upper_threshold": upper_threshold, # w1
#        "lower_threshold": lower_threshold # w2
#    }
    

#    return params

