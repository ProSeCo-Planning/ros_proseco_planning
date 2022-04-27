from typing import Optional, List

from .baseNonLinearReward import BaseNonLinearReward
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
    PREV_DIFF_VEL_VEL_DES = 7
    PREV_ABS_LANE_DIFF = 8
    PREV_DIFF_DES_LANE_CENT = 9


class NonLinearIrlReward(BaseNonLinearReward):
    """
    Class which specifies the NonLinearIrlReward function (in particular its features) which corresponds to the same class in the MCTs

    Instance variables:
        feature_names list -- list of the names of all used features in this reward function
        expert_cost_params list -- parameters of the reward function for the expert trajectories (reward for expert genertion is the linearIrlReward)
        initial_parameters list -- start parameters in the maximum likelihood procedure
        number_of_features int -- number of features in the reward function
        expert_trajectories_pickle_path string -- path for input and output of the expert trajectory messages, dependend if experts are generated or training is proceeded
        name string -- name of the reward
        mcts_cost_model_name -- name of the class inside the MCTS which mirrors the features
        T float -- parameter of the reward function that is normally equal to the episode length of the trajectories inside the IRL procedure (must be set by the irl model before training starts -  this is also asseterted in create_cost_message)
    """

    def __init__(
        self, expert_trajectories_pickle_path, expert_cost_params: Optional[List[float]]
    ):

        ####### IRL Reward settings ###########
        self.number_of_features = 10
        self.T = None
        self.expert_trajectories_pickle_path = expert_trajectories_pickle_path
        self.name = "NonLinearIrl"
        self.mcts_cost_model_name = "costNonLinear"

        ###### Only for expert creation - use linear cost model #########
        self.expert_feature_names = [
            "diff_vel_vel_des",
            "abs_lane_diff",
            "diff_des_lane_cent",
            "collided",
            "invalidState",
            "invalidAction",
            "accY",
        ]
        if expert_cost_params:
            self.expert_cost_params = np.array(
                expert_cost_params, ndmin=2
            ).T  # Transpose so that we get the same shape as linearIrl.initial_parameters
        else:
            self.expert_cost_params = None
        self.number_of_expert_features = len(self.expert_feature_names)
        self.expert_mcts_cost_model_name = "costLinear"

        # Expert parameters used for expert creation
        # sc1_single [1.0, 1.3, 2.5, -1.0, -1.3, -2.8, 0.4]
        # scen1 [1.0, 1.3, 1.9, -1.0, -1.0, -2.5, 0.3]
        # scen2 [2.0, 1.8, 1.9, -1.8, -1.8, -2.5, 0.3]
        # scen4 [1.0, 1.3, 1.9, -1.0, -1.0, -2.5, 0.3]
        # scen7 [1.0, 1.3, 1.9, -1.0, -1.0, -4.5, 0.3]
        # scen8 [1.0, 1.3, 1.9, -1.0, -1.0, -2.5, 0.3]
        # scen10 [1.4, 1.8, 1.9, -1.8, -1.8, -2.5, 0.5]

    ##### Getter methods #########

    def get_name(self):
        """
        return name of the reward model

        Return:
            string -- name of the reward model
        """
        return self.name

    def get_number_of_features(self):
        """
        return number of feaetures == number of input neurons in the reward neural net

        Return:
            int -- number of input features
        """
        return self.number_of_features

    def get_expert_parameters(self):
        """
        return the parameters of the expert reward model to create experts

        Return:
            np.array -- parameters of linear Irl reward to create expert trajectories with
        """
        return self.expert_cost_params

    def get_expert_pickle_folder(self):
        """
        return the path to the pickle files where the expert trajectory messages are saved or will be written to (important for training + expert creation)

        Return:
            string -- path where the pickle lie or should put to
        """
        return self.expert_trajectories_pickle_path

    def set_T(self, T):
        """
        sets the T parameter in the reward (often equal to the episode length therefore it can be setted from outside)

        Arguments:
            T float -- parameter T (often equal to episode length)
        """
        self.T = T

    def is_cooperative(self):
        return False

    ##### Main methods #############

    def features_to_vec(self, features, features_prev, meta_info):
        """Extracts features from the feature message of one agent in one timstep - this feature array is the input to the reward neural net

        Arguments:
            features ros_proseco_planning.msg.Features -- feature message for one agent in timestep t
            features_prev ros_proseco_planning.msg.Features -- feature message for one agent in timestep t-1
            meta_info ros_proseco_planning.msg.ScenarioInfo -- scenarioInfo - necessary to get for example lanewidth

        Returns:
            np.array -- feature values
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
        feature_vec[FeatureNames.PREV_DIFF_VEL_VEL_DES] = self.feature_vel_vel_des(
            features_prev.diff_vel_vel_des, features_prev.desired_vel
        )
        feature_vec[FeatureNames.PREV_ABS_LANE_DIFF] = self.feature_abs_lane_diff(
            features_prev.abs_lane_diff
        )
        feature_vec[
            FeatureNames.PREV_DIFF_DES_LANE_CENT
        ] = self.feature_diff_des_lane_cent(
            features_prev.diff_des_lane_cent, lane_width
        )

        return np.array(feature_vec)

    def create_cost_message(self, params):
        """Set current cost message which is retrieved to the slaves via the service - in this case the weights of the neural net are written to the cost message

        Arguments:
            params {np.array} -- cost parameter dict given from the non linear irl model and containing the weights of the reward neural net

        Return:
            msg.CostParam -- cost parameter message which is given to the slaves
        """
        assert not (self.T is None)
        w1 = params["w1"]
        w2 = params["w2"]
        b1 = np.squeeze(params["b1"])

        return {
            "name": [self.mcts_cost_model_name],
            "w1": w1,
            "w2": w2,
            "T": [self.T],
        }

        """
        cost_message = CostParam()
        cost_message.T = self.T
        cost_message.costModel = self.mcts_cost_model_name
        cost_message.w1 = self.convert_matrix_to_list(w1)
        cost_message.w1_nrows = np.shape(w1)[0]
        cost_message.w1_ncols = np.shape(w1)[1]
        cost_message.W2 = self.convert_matrix_to_list(w2)
        cost_message.w2_nrows = np.shape(w2)[0]
        cost_message.w2_ncols = np.shape(w2)[1]
        cost_message.b1 = b1
        cost_message.b1_nrows = np.shape(b1)[0]

        return cost_message
        """
        # TODO
        assert False

    def create_cost_message_expert(self, cost_params):
        """Sets expert cost message which is retrieved to the slaves via the service when experts are created - this is different to the normal cost message because it
            specifies a different cost function in the MCTS

        Arguments:
            params {np.array} -- cost parameter vector

        Return:
            json -- cost model which is given to the slaves for expert generation
        """

        return {
            "name": [self.expert_mcts_cost_model_name],
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
        }

    #### Feature methods ########

    def convert_matrix_to_list(self, W):
        """
        flattens the weight matrix of the weights of the reward neural net in order to put it in the cost param message

        Arguments:
            W np.array -- 2 dim np.array specifying a matrix

        Return:
            list -- list of values in the matrix in the order [row1,row2,...]
        """
        return W.flatten().tolist()

    def feature_acc_y(self, accy):
        """
        acceleration feature calculation

        Arguments:
            accy float -- integral over acc(y)^2 over route

        Return:
            float -- acc feature value
        """

        g = 9.81
        sqrt_acc = np.sqrt(0.5 * accy) / (0.25 * g)
        return max(1.0 - np.power(accy, 2.0), -1.0) / self.T

    def feature_vel_vel_des(self, d, desired_vel):
        """
        velocity feature calculation

        Arguments:
            d float -- difference of velocity and desired velocity
            desired_vel float -- value of desired velocity

        Return:
            float -- value of velocity feature
        """
        return max(1.0 - abs(d) / (abs(desired_vel) / 10.0), -1.0) / self.T

    def feature_abs_lane_diff(self, d):
        """
        absolute lane difference feature calculation

        Arguments:
            d float -- difference of desired lane number and current lane

        Return:
            float -- value of lane feature
        """
        return max(1.0 - abs(d), -1.0) / self.T

    def feature_diff_des_lane_cent(self, d, lane_width):
        """
        lane center feature calculation

        Arguments:
            d float -- differenc of the y position of the car and the center of its current lane
            lane_width float -- width of one lane

        Return:
            float -- value of lane center feature
        """
        return max(1.0 - abs(d) / (lane_width / 4.0), -1.0) / self.T

    def feature_invalid_action(self, is_invalid):
        """
        invalid action feature calculation

        Arguments:
            is_invalid bool -- boolean if action taken was invalid

        Return:
            float -- value of invalid action feature
        """
        if is_invalid:
            return 1.0 / self.T
        else:
            return 0.0
