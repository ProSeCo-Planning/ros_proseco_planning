from typing import Optional, List

from .baseLinearReward import BaseLinearReward
import numpy as np
import tensorflow.compat.v1 as tf

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
    DIFF_VEL_VEL_COOPERATIVE = 7
    ABS_LANE_DIFF_COOPERATIVE = 8
    DIFF_DES_LANE_CENT_COOPERATIVE = 9
    COLLIDED_COOPERATIVE = 10
    INVALID_STATE_COOPERATIVE = 11
    INVALID_ACTION_COOPERATIVE = 12
    ACC_Y_COOPERATIVE = 13


class LinearIrlRewardCooperative(BaseLinearReward):
    """
    Class which specifies the LinearIRL function (in particular its features) which corresponds to the same class in the MCTs

    Instance variables:
        feature_names list -- list of the names of all used features in this reward function
        expert_cost_params np.array -- parameters of the reward function for the expert trajectories
        number_of_features int -- number of features in the reward function
        expert_trajectories_pickle_path string -- path for input and output of the expert trajectory messages, dependend if experts are generated or training is proceeded
        name string -- name of the reward
        mcts_cost_model_name -- name of the cost class inside the MCTS which mirrors the features
        T float -- parameter of the reward function that is normally equal to the episode length of the trajectories inside the IRL procedure (must be set by the irl model before training starts - this is also asseterted in create_cost_message)
    """

    def __init__(
        self, expert_trajectories_pickle_path, expert_cost_params: Optional[List[float]]
    ):
        self.feature_names = [
            "diff_vel_vel_des",
            "abs_lane_diff",
            "diff_des_lane_cent",
            "collided",
            "invalidState",
            "invalidAction",
            "accY",
            "diff_vel_vel_des_cooperative",
            "abs_lane_diff_cooperative",
            "diff_des_lane_cent_cooperative",
            "collided_cooperative",
            "invalidState_cooperative",
            "invalidAction_cooperative",
            "accY_cooperative",
        ]
        cooperation_factor_experts = 0.0
        if expert_cost_params:
            self.expert_cost_params = np.array(
                expert_cost_params
                + [
                    cooperation_factor_experts * 0.0,
                    cooperation_factor_experts * 4.4,
                    cooperation_factor_experts * 0.0,
                    cooperation_factor_experts * -1.0,
                    cooperation_factor_experts * 0.0,
                    cooperation_factor_experts * 0.0,
                    cooperation_factor_experts * 0.0,
                ],
                ndmin=2,
            ).T  # Transpose so that we get the same shape as linearIrl.initial_parameters
        else:
            self.expert_cost_params = None

        self.number_of_features = len(self.feature_names)
        self.T = None
        self.expert_trajectories_pickle_path = expert_trajectories_pickle_path

        self.name = "LinearIrlCooperative"
        self.mcts_cost_model_name = "costLinearCooperative"

    def get_name(self):
        """
        return name of the reward model

        Return:
            string -- name of the reward model
        """
        return self.name

    def is_cooperative(self):
        return True

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
        assert not (feature_vec_other_agents is None)
        feature_vec = [0] * self.number_of_features

        # Ego features
        feature_vec[FeatureNames.DIFF_VEL_VEL_DES] = self.feature_vel_vel_des(
            features, meta_info
        )
        feature_vec[FeatureNames.ABS_LANE_DIFF] = self.feature_abs_lane_diff(
            features, meta_info
        )
        feature_vec[FeatureNames.DIFF_DES_LANE_CENT] = self.feature_diff_des_lane_cent(
            features, meta_info
        )
        feature_vec[FeatureNames.COLLIDED] = self.feature_collision(features, meta_info)
        feature_vec[FeatureNames.INVALID_STATE] = self.feature_invalid_state(
            features, meta_info
        )
        feature_vec[FeatureNames.INVALID_ACTION] = self.feature_invalid_action(
            features, meta_info
        )
        feature_vec[FeatureNames.ACC_Y] = self.feature_acc_y(features, meta_info)

        # Cooperative features
        feature_vec[
            FeatureNames.DIFF_VEL_VEL_COOPERATIVE
        ] = self.calculate_cooperative_feature(
            feature_vec_other_agents, meta_info, self.feature_vel_vel_des
        )
        feature_vec[
            FeatureNames.ABS_LANE_DIFF_COOPERATIVE
        ] = self.calculate_cooperative_feature(
            feature_vec_other_agents, meta_info, self.feature_abs_lane_diff
        )
        feature_vec[
            FeatureNames.DIFF_DES_LANE_CENT_COOPERATIVE
        ] = self.calculate_cooperative_feature(
            feature_vec_other_agents, meta_info, self.feature_diff_des_lane_cent
        )
        feature_vec[
            FeatureNames.COLLIDED_COOPERATIVE
        ] = self.calculate_cooperative_feature(
            feature_vec_other_agents, meta_info, self.feature_collision
        )
        feature_vec[
            FeatureNames.INVALID_STATE_COOPERATIVE
        ] = self.calculate_cooperative_feature(
            feature_vec_other_agents, meta_info, self.feature_invalid_state
        )
        feature_vec[
            FeatureNames.INVALID_ACTION_COOPERATIVE
        ] = self.calculate_cooperative_feature(
            feature_vec_other_agents, meta_info, self.feature_invalid_action
        )
        feature_vec[
            FeatureNames.ACC_Y_COOPERATIVE
        ] = self.calculate_cooperative_feature(
            feature_vec_other_agents, meta_info, self.feature_acc_y
        )

        return np.array(feature_vec)

    def create_cost_message(self, cost_params):
        """Set current cost message which is retrieved to the slaves via the service

        Arguments:
            params {np.array} -- cost parameter vector

        Return:
            msg.CostParam -- cost parameter message which is given to the slaves
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
            "w_acceleration_y_cooperative": cost_params[FeatureNames.ACC_Y_COOPERATIVE],
            "cost_collision_cooperative": cost_params[
                FeatureNames.COLLIDED_COOPERATIVE
            ],
            "cost_invalid_action_cooperative": cost_params[
                FeatureNames.INVALID_ACTION_COOPERATIVE
            ],
            "cost_invalid_state_cooperative": cost_params[
                FeatureNames.INVALID_STATE_COOPERATIVE
            ],
            "w_lane_center_deviation_cooperative": cost_params[
                FeatureNames.DIFF_DES_LANE_CENT_COOPERATIVE
            ],
            "w_lane_deviation_cooperative": cost_params[
                FeatureNames.ABS_LANE_DIFF_COOPERATIVE
            ],
            "w_velocity_deviation_cooperative": cost_params[
                FeatureNames.DIFF_VEL_VEL_COOPERATIVE
            ],
            "cost_enter_safe_range": [0.0],
            "reward_terminal": [0.0],
            "T": [self.T],
        }

    #### Feature methods ########

    def feature_acc_y(self, features, meta_info):
        """
        acceleration feature calculation

        Arguments:
            features msg.Feature - feature msg containing all features
            meta_info msg.ScenarioInfo - scenario info msg

        Return:
            float -- acc feature value
        """
        accy = features.accY
        g = 9.81
        sqrt_acc = np.sqrt(0.5 * accy) / (0.25 * g)
        return max(1.0 - np.power(accy, 2.0), -1.0) / self.T

    def feature_vel_vel_des(self, features, meta_info):
        """
        velocity feature calculation

        Arguments:
            features msg.Feature - feature msg containing all features
            meta_info msg.ScenarioInfo - scenario info msg

        Return:
            float -- value of velocity feature
        """
        d = features.diff_vel_vel_des
        desired_vel = features.desired_vel
        return max(1.0 - abs(d) / (abs(desired_vel) / 10.0), -1.0) / self.T

    def feature_abs_lane_diff(self, features, meta_info):
        """
        absolute lane difference feature calculation

        Arguments:
            features msg.Feature - feature msg containing all features
            meta_info msg.ScenarioInfo - scenario info msg

        Return:
            float -- value of lane feature
        """
        d = features.abs_lane_diff
        return max(1.0 - abs(d), -1.0) / self.T

    def feature_diff_des_lane_cent(self, features, meta_info):
        """
        lane center feature calculation

        Arguments:
            features msg.Feature - feature msg containing all features
            meta_info msg.ScenarioInfo - scenario info msg

        Return:
            float -- value of lane center feature
        """
        d = features.diff_des_lane_cent
        lane_width = meta_info.laneWidth
        return max(1.0 - abs(d) / (lane_width / 4.0), -1.0) / self.T

    def feature_invalid_action(self, features, meta_info):
        """
        invalid action feature calculation

        Arguments:
            features msg.Feature - feature msg containing all features
            meta_info msg.ScenarioInfo - scenario info msg

        Return:
            float -- value of invalid action feature
        """
        is_invalid = features.invalidAction
        if is_invalid:
            return 1.0 / self.T
        else:
            return 0.0

    def feature_invalid_state(self, features, meta_info):
        """
        invalid state feature calc

        Arguments:
            features msg.Feature - feature msg containing all features
            meta_info msg.ScenarioInfo - scenario info msg

        Return:
            float -- value of invalid state feature
        """
        if features.invalidState:
            return 1.0
        else:
            return 0.0

    def feature_collision(self, features, meta_info):
        """
        collision feature calc

        Arguments:
            features msg.Feature - feature msg containing all features
            meta_info msg.ScenarioInfo - scenario info msg

        Return:
            float -- value of collision feature
        """
        if features.collided:
            return 1.0
        else:
            return 0.0

    def calculate_cooperative_feature(
        self, feature_vec_other_agents, meta_info, feature_function
    ):
        """
        calculates the cooperative feature values by iterating over the feature message of the other agents

        Arguments:
            feature_vec_other_agents [msg.Feature] -- list of the feature msg of the other agents in the trajectory
            meta_info msg.ScenarioInfo -- scenario info msg
            feature_function func -- feature instance method for which the cooperative feature should be calculated (self.feature_collision for example)
        """
        number_of_other_agents = len(feature_vec_other_agents)
        feature_sum = 0.0
        for features in feature_vec_other_agents:
            feature_sum += feature_function(features, meta_info)
        return feature_sum / float(number_of_other_agents)
