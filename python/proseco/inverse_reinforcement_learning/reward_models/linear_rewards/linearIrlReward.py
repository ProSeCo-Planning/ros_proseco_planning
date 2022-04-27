from typing import List, Optional

from .baseLinearReward import BaseLinearReward
import numpy as np
import tensorflow.compat.v1 as tf
import math

from proseco.utility.constants import GRAVITY, ACTION_DURATION

tf.disable_v2_behavior()


class FeatureNames:
    """
    Enum defining the feature indexes in the feature vec
    """

    DIFF_VEL_VEL_DES = 0
    ABS_LANE_DIFF = 1
    DIFF_DES_LANE_CENT = 2
    COLLIDED = 3
    INVALID_STATE = 4
    INVALID_ACTION = 5
    ACC_Y = 6


class LinearIrlReward(BaseLinearReward):
    """
    Class which specifies the LinearIRL function (in particular its features) which corresponds to the same class in the MCTS

    Instance variables:
        feature_names list -- list of the names of all used features in this reward function
        expert_cost_params np.array -- parameters of the reward function for the expert trajectories
        number_of_features int -- number of features in the reward function
        expert_trajectories_pickle_path string -- path for input and output of the expert trajectory messages, dependend if experts are generated or training is proceeded
        name string -- name of the reward
        mcts_cost_model_name -- name of the cost class inside the MCTS which mirrors the features
        T float -- parameter of the reward function that is normally equal to the episode length of the trajectories inside the IRL procedure (must be set by the irl model before training starts - this is also asserted in create_cost_message)
    """

    def __init__(
        self,
        expert_trajectories_pickle_path: str,
        expert_cost_params: Optional[List[float]],
    ):
        self.name = "LinearIrl"
        self.mcts_cost_model_name = "costLinear"
        self.expert_trajectories_pickle_path = expert_trajectories_pickle_path
        self.T = None
        self.feature_names = [
            "diff_vel_vel_des",
            "abs_lane_diff",
            "diff_des_lane_cent",
            "collided",
            "invalidState",
            "invalidAction",
            "accY",
        ]
        self.number_of_features = len(self.feature_names)

        if expert_cost_params:
            self.expert_cost_params = np.array(
                expert_cost_params, ndmin=2
            ).T  # Transpose so that we get the same shape as linearIrl.initial_parameters
        else:
            self.expert_cost_params = None

    def get_name(self):
        """
        return name of the reward model

        Return:
            string -- name of the reward model
        """
        return self.name

    def is_cooperative(self):
        return False

    def get_feature_names(self):
        """
        returns the names of the features

        Return:
            list -- names of features
        """
        return self.feature_names

    def get_number_of_features(self):
        """
        return number of features == number of input neurons in the reward neural net

        Return:
            int -- number of input features
        """
        return self.number_of_features

    def get_expert_parameters(self):
        """
        return the parameters of the expert reward model to create experts

        Return:
            np.array -- parameters create expert trajectories with
        """
        return self.expert_cost_params

    def get_expert_pickle_folder(self):
        """
        return the path to the pickle files where the expert trajectory messages are saved or will be written to (important for training + expert creation)

        Return:
            string -- path where the pickle lie or should put to
        """
        return self.expert_trajectories_pickle_path

    ##### Main methods #############

    def features_to_vec(
        self, features, features_prev, meta_info, feature_vec_other_agents
    ):
        """Extracts features from the feature message of one agent in one timstep

        Arguments:
            features ros_proseco_planning.msg.Features -- feature message for one agent in timestep t
            features_prev ros_proseco_planning.msg.Features -- feature message for one agent in timestep t-1
            meta_info ros_proseco_planning.msg.ScenarioInfo -- scenarioInfo - necessary to get for example lanewidth

        Returns:
            np.array -- feature valus in list
        """

        feature_vec = [0] * self.number_of_features
        lane_width = meta_info.laneWidth

        feature_vec[FeatureNames.DIFF_VEL_VEL_DES] = self.feature_vel_vel_des(
            features.diff_vel_vel_des, features.desired_vel
        )
        feature_vec[FeatureNames.ABS_LANE_DIFF] = self.feature_abs_lane_diff(
            features.abs_lane_diff
        )
        feature_vec[FeatureNames.DIFF_DES_LANE_CENT] = self.feature_diff_des_lane_cent(
            features.diff_des_lane_cent, lane_width
        )
        feature_vec[FeatureNames.COLLIDED] = int(features.collided)
        feature_vec[FeatureNames.INVALID_STATE] = int(features.invalidState)
        feature_vec[FeatureNames.INVALID_ACTION] = self.feature_invalid_action(
            features.invalidAction
        )
        feature_vec[FeatureNames.ACC_Y] = self.feature_acc_y(features.accY)
        return np.array(feature_vec)

    def create_cost_message(self, cost_params):
        """Set current cost message which is retrieved to the slaves via the service

        Arguments:
            params {np.array} -- cost parameter vector

        Return:
            json -- cost model
        """

        return {
            "name": [self.mcts_cost_model_name],
            "w_acceleration_x": [0.0],
            "w_acceleration_y": cost_params[FeatureNames.ACC_Y],
            "cost_collision": cost_params[FeatureNames.COLLIDED],
            "cost_invalid_action": cost_params[FeatureNames.INVALID_ACTION],
            "cost_invalid_state": cost_params[FeatureNames.INVALID_STATE],
            "w_lane_center_deviation": cost_params[FeatureNames.DIFF_DES_LANE_CENT],
            "w_lane_change": [0.0],
            "w_lane_deviation": cost_params[FeatureNames.ABS_LANE_DIFF],
            "w_velocity_deviation": cost_params[FeatureNames.DIFF_VEL_VEL_DES],
            "cost_enter_safe_range": [0.0],
            "reward_terminal": [0.0],
            "T": [self.T],
        }

    def feature_acc_y(self, cumulated_squared_acceleration: float) -> float:
        """Calculates the acceleration y feature.

        Parameters
        ----------
        cumulated_squared_acceleration : float
            The Integral of the squared acceleration in y direction.

        Returns
        -------
        float
            The value of the acceleration y feature.
        """
        assert cumulated_squared_acceleration >= 0
        g_y = math.sqrt(cumulated_squared_acceleration / ACTION_DURATION) / GRAVITY
        return max(1.0 - 4 * g_y, -1.0) / self.T

    def feature_vel_vel_des(
        self, velocity_deviation: float, desired_vel: float
    ) -> float:
        """Calculates the velocity deviation feature.

        Parameters
        ----------
        velocity_deviation : float
            The difference between the current velocity and the desired velocity.
        desired_vel : float
            The desired velocity.

        Returns
        -------
        float
            The value of the velocity deviation feature.
        """
        return (
            max(1.0 - abs(velocity_deviation) / (abs(desired_vel) / 10.0), -1.0)
            / self.T
        )

    def feature_abs_lane_diff(self, lane_deviation: int) -> float:
        """Calculates the lane deviation feature.

        Parameters
        ----------
        lane_deviation : int
            The difference between the current lane and the desired lane.

        Returns
        -------
        float
            The value of the lane deviation feature.
        """
        return max(1.0 - abs(lane_deviation), -1.0) / self.T

    def feature_diff_des_lane_cent(self, d: float, lane_width: float) -> float:
        """Calculates the desired lane center feature. The feature is positive as long as the deviation is smaller than a quarter of the lane width.

        Parameters
        ----------
        d : float
            The difference between the y position and the center of the lane.
        lane_width : [type]
            The width of the lane.

        Returns
        -------
        float
            The value of the desired lane center feature.
        """
        return max(1.0 - abs(d) / (lane_width / 4.0), -1.0) / self.T

    def feature_invalid_action(self, is_invalid) -> float:
        """Calculates the invalid action feature.

        Parameters
        ----------
        is_invalid : bool
            Whether the action is invalid or not.

        Returns
        -------
        float
            The value of the invalid action feature.
        """
        if is_invalid:
            return 1.0 / self.T
        else:
            return 0.0
