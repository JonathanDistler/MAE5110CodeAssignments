# animate.py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


def animate_rimless_wheel(time_traj, state_traj, params, fps=30, playback_speed=1.0):
    #Animate the rimless wheel rolling downhill.
    l = params["length"]
    gamma = params["slope_angle"]
    alpha = params["half_spoke_angle"]
    n_spokes = int(round(np.pi / alpha))

    theta_traj = state_traj[0, :]

    # detect resets: theta jumps down by ~2*alpha at each collision
    dtheta = np.diff(theta_traj)
    collision_mask = dtheta < -alpha
    step_count = np.concatenate(([0], np.cumsum(collision_mask)))

    step_length = 2 * l * np.sin(alpha)
    downhill_dir = np.array([np.cos(gamma), -np.sin(gamma)])

    foot_traj = step_count[np.newaxis, :] * step_length * downhill_dir[:, np.newaxis]
    hub_traj = foot_traj + l * np.vstack([np.sin(theta_traj), np.cos(theta_traj)])

    # subsample to a manageable number of animation frames
    playback_dt = playback_speed / fps
    frame_times = np.arange(time_traj[0], time_traj[-1], playback_dt)
    frame_idx = np.searchsorted(time_traj, frame_times)
    frame_idx = frame_idx[frame_idx < len(time_traj)]

    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    x_min, x_max = hub_traj[0].min() - l, hub_traj[0].max() + l
    y_min, y_max = hub_traj[1].min() - l, hub_traj[1].max() + l
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    slope_x = np.array([x_min, x_max])
    slope_y = -slope_x * np.tan(gamma)
    ax.plot(slope_x, slope_y, "k-", linewidth=1)

    hub_point, = ax.plot([], [], "ko", markersize=5)
    spoke_lines = [ax.plot([], [], "b-", linewidth=2)[0] for _ in range(n_spokes)]
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def update(frame):
        i = frame_idx[frame]
        hub = hub_traj[:, i]
        theta = theta_traj[i]

        hub_point.set_data([hub[0]], [hub[1]])
        for k, line in enumerate(spoke_lines):
            spoke_angle = theta + 2 * np.pi * k / n_spokes
            tip = hub + l * np.array([np.sin(spoke_angle), np.cos(spoke_angle)])
            line.set_data([hub[0], tip[0]], [hub[1], tip[1]])

        time_text.set_text(f"t = {time_traj[i]:.2f} s")
        return [hub_point, time_text] + spoke_lines

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_idx), interval=1000 / fps, blit=True
    )
    plt.show()
    return anim