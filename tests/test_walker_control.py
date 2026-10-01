"""Exercise assignment functions without running its plots and grid searches"""
import ast
from pathlib import Path

import numpy as np
import pytest

from models import inverted_pendulum_walker as model


@pytest.fixture(scope="module")
def controller():
    path = Path(__file__).resolve().parents[1] / "assignment_2.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    definitions = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef)],
        type_ignores=[],
    )
    namespace = {"np": np, "model": model}
    exec(compile(definitions, str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("side", [-1, 1])
def test_start_angle_above_inclined_plane(side):
    params = model.generate_params()
    boundary = params["incline"] + side * np.pi / 2
    model.validate_initial_condition([boundary, 0.0], params)
    with pytest.raises(ValueError, match="below the ground plane"):
        model.validate_initial_condition([boundary + side * 0.01, 0.0], params)


def test_bounded_torque_controller_settles_inside_roa(controller):
    params = model.generate_params()
    lower_torque, upper_torque = controller["get_control_bounds"](params)[2:]
    # Both motion directions, nonzero angles, and points near capture bounds
    for theta in [-0.04, 0.0, 0.08]:
        lower, upper = controller["capture_bounds"](theta, params)
        for fraction in [0.05, 0.5, 0.95]:
            state = np.array([theta, lower + fraction * (upper - lower)])
            assert controller["state_is_in_roa"](state, params)
            torque = controller["feedback_linearization_controller"](state, params)
            assert lower_torque <= torque <= upper_torque
            history = np.asarray(controller["simulate_balance"](state, params, timestep=0.001))
            assert np.all(np.isfinite(history))
            assert np.max(np.abs(history[-1])) < 1e-6
            torques = np.array([
                controller["feedback_linearization_controller"](point, params)
                for point in history
            ])
            assert np.all((torques >= lower_torque) & (torques <= upper_torque))
            assert np.all(np.abs(history[:, 0] - params["incline"]) <= np.pi / 2)
