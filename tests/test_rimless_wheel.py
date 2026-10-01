import numpy as np
import pytest

from models import compass_gait, inverted_pendulum_walker, rimless_wheel


def test_default_spokes_clear_ground():
    params = rimless_wheel.generate_params()
    _, tips = rimless_wheel.calculate_positions(
        rimless_wheel.generate_initial_condition(params), params
    )
    normal = [np.sin(params["incline"]), np.cos(params["incline"])]
    assert np.all(tips @ normal >= -1e-12)


@pytest.mark.parametrize("direction", [-1, 1])
def test_spoke_collision(direction):
    params = rimless_wheel.generate_params()
    alpha = np.pi / params["N_spokes"]
    boundary = params["incline"] + direction * alpha
    initial = np.array([boundary, direction * 1.0])
    state, contacts, displacement = rimless_wheel.advance_step(0, initial, 0.001, params)
    assert contacts == 1
    assert direction * displacement[0] > 0
    assert abs(state[0] - params["incline"]) < alpha
    np.testing.assert_allclose(displacement[1],
                               -np.tan(params["incline"]) * displacement[0], atol=1e-12)


@pytest.mark.parametrize("model", [rimless_wheel, inverted_pendulum_walker])
def test_reject_initial_leg_penetration(model):
    params = model.generate_params()
    limit = np.pi / params["N_spokes"] if model is rimless_wheel else np.pi / 2
    angle = params["incline"] + limit + 0.01
    with pytest.raises(ValueError, match="ground plane"):
        model.validate_initial_condition([angle, 0.0], params)


def test_compass_initial_angles_checked_geometrically():
    params = compass_gait.generate_params()
    compass_gait.validate_initial_condition(compass_gait.generate_initial_condition(), params)
    with pytest.raises(ValueError, match="ground plane"):
        compass_gait.validate_initial_condition([0.4, 0.0, 0.0, 0.0], params)
