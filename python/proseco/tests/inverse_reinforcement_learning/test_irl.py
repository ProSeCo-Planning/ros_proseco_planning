import pytest
import math
from pytest import approx
from proseco.inverse_reinforcement_learning.reward_models.linear_rewards.linearIrlReward import (
    LinearIrlReward,
)

from proseco.utility.constants import GRAVITY, ACTION_DURATION


@pytest.fixture
def linearIrlReward():
    return LinearIrlReward("", [])


def test_linearIRLReward_features(linearIrlReward):
    linearIrlReward.T = 10
    assert linearIrlReward.feature_invalid_action(True) == 1.0 / 10.0
    assert linearIrlReward.feature_invalid_action(False) == 0.0

    assert linearIrlReward.feature_diff_des_lane_cent(0.0, 3.5) == approx(1.0 / 10.0)
    assert linearIrlReward.feature_diff_des_lane_cent(3.5 / 4, 3.5) == approx(0.0)
    assert linearIrlReward.feature_diff_des_lane_cent(3.5 / 2, 3.5) == approx(
        -1.0 / 10.0
    )

    assert linearIrlReward.feature_abs_lane_diff(0) == approx(1.0 / 10.0)
    assert linearIrlReward.feature_abs_lane_diff(1) == approx(0.0)
    assert linearIrlReward.feature_abs_lane_diff(2) == approx(-1.0 / 10.0)

    assert linearIrlReward.feature_vel_vel_des(0.0, 10.0) == approx(1.0 / 10.0)
    assert linearIrlReward.feature_vel_vel_des(1.0, 10.0) == approx(0.0)
    assert linearIrlReward.feature_vel_vel_des(2.0, 10.0) == approx(-1.0 / 10.0)

    assert linearIrlReward.feature_vel_vel_des(0.0, 10.0) == approx(1.0 / 10.0)
    assert linearIrlReward.feature_vel_vel_des(1.0, 10.0) == approx(0.0)
    assert linearIrlReward.feature_vel_vel_des(2.0, 10.0) == approx(-1.0 / 10.0)

    assert linearIrlReward.feature_acc_y(0.0) == approx(1.0 / 10.0)
    assert linearIrlReward.feature_acc_y(
        math.pow(GRAVITY / 4, 2) * ACTION_DURATION
    ) == approx(0.0)
    assert linearIrlReward.feature_acc_y(
        4 * math.pow(GRAVITY / 4, 2) * ACTION_DURATION
    ) == approx(-1.0 / 10.0)
