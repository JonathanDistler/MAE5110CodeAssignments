import numpy as np

from models import inverted_pendulum_walker as walker


def test_initial_pose_is_upright():
    state = walker.generate_initial_condition()
    assert state[0] == 0.0
    assert state[1] > 0.0


def test_forward_touchdown_resets_and_advances_stance():
    params = walker.generate_params()
    boundary = params["incline"] + params["angle_of_attack"]
    state = np.array([boundary - 0.001, 1.0])
    next_state, contacts, displacement = walker.advance_step(0.0, state, 0.002, params)
    assert contacts == 1
    assert next_state[0] < 0.0
    assert 0.0 < next_state[1] < state[1]
    assert displacement[0] > 0.0
    np.testing.assert_allclose(
        displacement[1], -np.tan(params["incline"]) * displacement[0], atol=1e-12
    )
    # Changing stance feet preserves the hip position at the instant of impact.
    post_angle = boundary - 2 * params["angle_of_attack"]
    before = params["length"] * np.array([np.sin(boundary), np.cos(boundary)])
    after = displacement + params["length"] * np.array([
        np.sin(post_angle), np.cos(post_angle)
    ])
    np.testing.assert_allclose(before, after, atol=1e-12)
