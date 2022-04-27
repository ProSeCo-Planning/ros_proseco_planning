import json
import logging
import os
from typing import List, Tuple, Optional

import numpy as np
from proseco.utility.io import save_data
import tensorflow.compat.v1 as tf

from proseco.inverse_reinforcement_learning.reward_models.linear_rewards.baseLinearReward import (
    BaseLinearReward,
)
from proseco.inverse_reinforcement_learning.sampling_policies.baseSamplingPolicy import (
    BaseSamplingPolicy,
)
from proseco.inverse_reinforcement_learning.trajectory import (
    EvaluationRun,
    Trajectory,
    ScenarioInfo,
)
from . import utils
from .baseIrlModel import BaseIRLModel

from proseco.utility.ui import get_log_level, get_logger

logger = get_logger("ProSeCo IRL", create_handler=False)

tf.disable_v2_behavior()


class LinearIRLModel(BaseIRLModel):
    def __init__(
        self,
        reward_model: BaseLinearReward,
        sampling_policy: BaseSamplingPolicy,
        learning_rate,
        number_of_steps,
        training_name,
        optimizer_type,
        work_dir,
    ):
        """

        Parameters
        ----------
        reward_model
        sampling_policy
        learning_rate
        number_of_steps
        training_name
        optimizer_type
        work_dir
        """
        self.use_random_initial_parameters = True
        self.initial_parameters_magnitute = 0.2
        self.updates_inner_loop = 1

        super(LinearIRLModel, self).__init__(
            reward_model,
            sampling_policy,
            optimizer_type,
            learning_rate,
            number_of_steps,
            training_name,
            work_dir,
        )

    ### Initialization + Getter + Setter ############

    def initialize(self, irl_config=None):
        self.create_paths()
        if irl_config:
            self.save_config(irl_config)
        self.set_tensorboard_path()
        self.initialize_parameters()
        self.train_counter = 0

    def initialize_parameters(self):
        """
        Initializes the reward parameters either with random values or with zeros.
        """
        if self.use_random_initial_parameters:
            self.initial_parameters = self.create_random_parameters()
        else:
            self.initial_parameters = np.zeros((self.number_of_features, 1))

        logger.info(
            f"Initial parameters: \n{self.feature_names}\n{self.initial_parameters}"
        )

    def create_random_parameters(self) -> np.ndarray:
        """Creates random parameters for the reward model

        Returns
        -------
        np.ndarray
            The random parameters.
        """
        return np.random.rand(
            self.number_of_features, 1
        ) * self.initial_parameters_magnitute - (self.initial_parameters_magnitute / 2)

    ###### Training Methods #########

    def log_likelihood_with_importance_sampling(self, sample_rewards) -> None:
        """
        Calculates the log likelihood of the current parameters with importance sampling

        e^R(samples_i) / P(samples_i)
        """
        self.sum_elements = tf.multiply(
            tf.math.reciprocal(self.likelihoods_samples_ph),
            tf.math.exp(sample_rewards),
        )

    def log_likelihood_without_importance_sampling(self, sample_rewards) -> None:
        """
        Calculates the log likelihood of the current parameters without importance sampling
        """
        self.sum_elements = (
            tf.math.exp(sample_rewards)
            / self.sampling_policy.get_artificial_importance_weight()
        )

    def build_model(self):
        """
        Builds the model for maximum likelihood of a linear return function given the Maximum Entropy model.

        Input tensors:
            expert_features_ph      -- placeholder for expert features
            sampled_features_ph     -- placeholder for features of sampled trajectories
            likelihood_samples_ph   -- placeholder for likelihood values (importance weights of samples)

        Variables:
            parameters --   tensorflow variable which gets updated by the optimizer

        Target tensor:
            loglikeli --    log likelihood (estimate) of the expert trajectories and the current reward parameters

        Optimizer:
            optimizer --    optimizer which minimizes the negative log likelihood (maximizes loglikeli)
        """

        # model input
        self.expert_features_ph = tf.placeholder(
            tf.float64, shape=[None, self.number_of_features]
        )
        self.sampled_features_ph = tf.placeholder(
            tf.float64, shape=[None, self.number_of_features]
        )
        self.likelihoods_samples_ph = tf.placeholder(tf.float64, shape=[None, 1])

        # reward parameters
        self.parameters = tf.Variable(
            tf.constant(
                self.initial_parameters,
                shape=[self.number_of_features, 1],
                dtype=tf.float64,
            ),
            name="parameters",
            dtype=tf.float64,
        )

        # calculating loglikeli
        expert_rewards = tf.matmul(self.expert_features_ph, self.parameters)
        sample_rewards = tf.matmul(self.sampled_features_ph, self.parameters)

        if self.sampling_policy.do_drop_importance_weights():
            self.log_likelihood_without_importance_sampling(sample_rewards)
        else:
            self.log_likelihood_with_importance_sampling(sample_rewards)

        self.loglikeli = tf.math.reduce_mean(expert_rewards) - tf.math.log(
            tf.math.reduce_mean(self.sum_elements)
        )

        ## optimizer gets chosen
        optimizer = self.create_optimizer()

        # minimize
        self.grads = optimizer.compute_gradients(-1 * self.loglikeli)
        grad_values, _ = self.grads[0]
        self.grad_norm = tf.math.sqrt(tf.math.reduce_sum(tf.math.pow(grad_values, 2)))
        tf.summary.scalar(name="train/grad_norm", tensor=self.grad_norm)
        self.step = optimizer.apply_gradients(self.grads)

        # self.step = optimizer.minimize(-1*self.loglikeli)

        self.init_op = tf.variables_initializer(tf.global_variables())

        self.add_tensorboard_summaries()

        self.initialize_tf_session()

    def make_update(self, trajectory_list: List[EvaluationRun]) -> Tuple[dict, dict]:
        """
        Main method which gets called by trainingEngine when stack is full

        Procedure:
            1. retrieves feature counts from trajectory list
            2. makes parameter update
            3. adjusts noise parameters (calling sampling_policy)
            4. creates new parameter and noise messages and returns them


        Parameters
        ----------
        trajectory_list
            list of trajectory message coming from the current stack

        Returns
        -------
        CostParam, NoiseParam
            new cost and noise message for service in trainingEngine after parameter update
        """
        self.assert_validity(trajectory_list)
        (
            samples_features,
            likelihood_tensor,
            average_selection_likelis,
            meta_info_trajectories,
        ) = self.get_trajectory_tensors_from_list(trajectory_list)

        if get_log_level("ProSeCo IRL") == logging.DEBUG:
            utils.show_feature_logs(
                samples_features, self.expert_features, self.feature_names
            )
        new_parameters = self.make_parameters_update(
            samples_features, likelihood_tensor, meta_info_trajectories
        )

        return self.create_parameter_messages(new_parameters, expert=False)

    def make_parameters_update(
        self,
        samples_tensor: np.ndarray,
        likelihood_tensor: np.ndarray,
        meta_info_trajectories,
    ) -> np.ndarray:
        """
        Main gradient update method - gets a sample feature count with likelis and makes a gradient descent update on the neg. loglikeli

        Parameters
        ----------
        samples_tensor
            numpy array with shape [number_of_trajectories,number_of_features] containing all featurecounts for the samples trajectories
        likelihood_tensor
            numpy array with shape [number_of_trajectories,1] containing the likelihoods of every trajectory under the current importance sampling policy
        meta_info_trajectories

        Returns
        -------
        np.ndarray
            updated cost parameters after training step
        """

        logger.info(f"Update step: {self.train_counter}/{self.number_of_steps-1}")

        parameter_vec_old = self.parameters.eval(session=self.sess)
        logger.debug(f"Parameters before update: {parameter_vec_old}")
        train_summary = utils.get_train_metrics(
            samples_tensor, parameter_vec_old, self.expert_features, self.feature_names
        )
        logger.debug(f"Likelihoods: {likelihood_tensor}")

        for _ in range(self.updates_inner_loop):
            (
                merge_summary,
                _,
                gradient,
                gradient_norm,
                sum_elem,
            ) = self.run_update_session(
                self.expert_features, samples_tensor, likelihood_tensor
            )
            logger.debug(f"Gradient: {gradient}")
            logger.debug(f"Gradient norm: {gradient_norm}")

        ### Outputs start ###
        weight_of_sum_elem = sum_elem / np.sum(sum_elem)
        logger.debug(f"Weight of sum elements: {weight_of_sum_elem}")
        parameter_vec_new = self.parameters.eval(session=self.sess)
        logger.debug(f"Parameters after update: {parameter_vec_new}")

        if self.train_counter == 1:
            self.add_experiment_metadata(
                self.expert_features.shape[0], samples_tensor.shape[0]
            )
        ### add logs to tensorboard
        self.add_scalar_to_tensorboard(
            np.max(weight_of_sum_elem), "max_proposal/weight"
        )

        logger.debug("Information of sample with highest proposal weight")
        max_weight_index = np.argmax(weight_of_sum_elem)
        agent_ids, scenario_names = meta_info_trajectories
        logger.debug(f"Agent ids: {agent_ids}")
        agent_id_max_weight = agent_ids[max_weight_index]
        logger.debug(f"Agent with max weight: {agent_id_max_weight}")
        self.add_scalar_to_tensorboard(
            agent_id_max_weight, "max_proposal/agent_id_max_weight"
        )
        logger.debug(f"Samples tensor max weight: {samples_tensor[max_weight_index]}")
        max_weight_scenario_name = scenario_names[max_weight_index]
        logger.debug(f"Scenario with max weight: {max_weight_scenario_name}")
        scenario_in_list, scenario_no = utils.get_scenario_number_for_tag(
            max_weight_scenario_name
        )
        if scenario_in_list:
            self.add_scalar_to_tensorboard(scenario_no, "max_proposal/scenario_no")
        proposal_summary = utils.create_generic_feature_diff_summary(
            "max_proposal/",
            np.expand_dims(samples_tensor[max_weight_index], axis=0),
            self.expert_features,
            self.feature_names,
        )
        ### Outputs end ###

        if np.isnan(parameter_vec_new).any():
            parameter_vec_new = parameter_vec_old
            assign_op = self.parameters.assign(parameter_vec_old)
            self.sess.run(assign_op)
            logger.warning(
                f"Nan appeared in parameters: {parameter_vec_new} repeating update step"
            )
        else:
            self.train_writer.add_summary(train_summary, self.train_counter)
            self.train_writer.add_summary(merge_summary, self.train_counter)
            self.train_writer.add_summary(proposal_summary, self.train_counter)
            self.train_counter += 1

        return parameter_vec_new

    def save_weights(self, save_path):
        parameters = self.parameters.eval(session=self.sess)
        data = {
            k: v
            for k, v in zip(self.reward_model.get_feature_names(), parameters.tolist())
        }
        save_data(data, os.path.join(save_path, "parameters.json"), sort=False)

    ##### Trajectory preprocessing ########

    def count_features_in_trajectory(
        self, trajectory_message: Trajectory, scenarioInfo: ScenarioInfo
    ) -> np.ndarray:
        """
        Counts the features from one trajectory by iterating over all time steps and all agents

        Parameters
        ----------
        trajectory_message
            one sampled trajectory
        scenarioInfo
            info about the scenario.

        Returns
        -------
        np.ndarray
            counted features over the complete trajectory (len=9)
        """

        # initialize new feature vec
        feature_vec = np.zeros(self.number_of_features, dtype=float)
        previous_state_action = trajectory_message.initialState
        counter = 0

        for state_action_pair in trajectory_message.trajectory:
            prev_agent_vec = previous_state_action
            agent_vec = state_action_pair

            # cumulate features over trajectory over all agents
            ## Need to check if featureOtherAgents exist because this part of the message was added later and a lot of expert pickle files do not have this attribute (also only used by cooperative reward)
            if hasattr(agent_vec, "featuresOtherAgents"):
                feature_vec = feature_vec + self.reward_model.features_to_vec(
                    agent_vec.features,
                    prev_agent_vec.features,
                    scenarioInfo,
                    agent_vec.featuresOtherAgents,
                )
            else:
                assert not self.reward_model.is_cooperative()
                feature_vec = feature_vec + self.reward_model.features_to_vec(
                    agent_vec.features,
                    prev_agent_vec.features,
                    scenarioInfo,
                    None,
                )

            previous_state_action = state_action_pair
            counter += 1

        self.compare_to_episode_length(counter)
        logger.debug(f"Feature vector: {feature_vec}")
        return feature_vec

    def get_feature_tensor_from_list(
        self, trajectory_list: List[EvaluationRun]
    ) -> np.ndarray:
        """Get the feature tensor from a "trajectory list" (list of evaluation runs).

        Note
        ----
        The returned tensor corresponds to the first tuple element returned by the method `get_trajectory_tensors_from_list`.

        Parameters
        ----------
        trajectory_list : List[EvaluationRun]
            list of evaluation runs

        Returns
        -------
        np.ndarray
            feature tensor
        """
        features: List[np.ndarray] = []
        # iterates over all single agent trajectory messages and creates the feature vectors
        for trajectory_message, scenarioInfo in self.split_trajectory_list(
            trajectory_list
        ):
            feature_vec = self.count_features_in_trajectory(
                trajectory_message, scenarioInfo
            )
            features.append(np.expand_dims(feature_vec, axis=0))

        return np.concatenate(features, axis=0)

    def get_trajectory_tensors_from_list(
        self, trajectory_list: List[EvaluationRun]
    ) -> Tuple[np.ndarray, np.ndarray, List[float], Optional[np.ndarray]]:
        """
        Summarizes all the trajectory messages on the stack in a feature count tensor and a likelihood tensor

        Parameters
        ----------
        trajectory_list
            trajectory list

        Returns
        -------
        np.ndarray
            tensor of feature counts
        np.ndarray
            tensor of likelis of trajectories
        List[float]
            tensor of average selection likelis
        Optional[np.ndarray]
            tensor of meta info trajectories
        """

        samples = []
        likelis = []
        average_selection_likelis = []
        agent_ids = []
        scenario_names = []

        for trajectory_message, scenarioInfo in self.split_trajectory_list(
            trajectory_list
        ):
            feature_vec = self.count_features_in_trajectory(
                trajectory_message, scenarioInfo
            )
            (
                likelihood,
                average_selection_likelihood,
            ) = self.extract_trajectory_likelihood(trajectory_message)
            samples.append(np.expand_dims(feature_vec, axis=0))
            likelis.append(np.expand_dims(likelihood, axis=0))
            average_selection_likelis.append(average_selection_likelihood)
            agent_id, scenario_name = self.extract_information_from_trajectory_message(
                trajectory_message, scenarioInfo
            )
            scenario_names.append(scenario_name)
            agent_ids.append(agent_id)

            # Concatenate all feature vec for all num_samples trajectory to one np array
        samples_tensor = np.concatenate(samples, axis=0)
        likelihood_tensor = np.concatenate(likelis, axis=0)
        meta_info_trajectories = agent_ids, scenario_names
        return (
            samples_tensor,
            likelihood_tensor,
            average_selection_likelis,
            meta_info_trajectories,
        )

    ### Tensorboard # Logging methods ######

    def extract_information_from_trajectory_message(
        self, trajectory_message: Trajectory, scenarioInfo: ScenarioInfo
    ):
        """
        For logging/debugging: extracts some meta info about a generated trajectory like the id of the agent and the
        scenario name

        Parameters
        ----------
        trajectory_message
            one sampled trajectory
        scenarioInfo
            info about the scenario.

        Returns
        -------
        int
            agent id
        string
            scenario name
        """
        return trajectory_message.agentId, scenarioInfo.scenarioName

    def add_validation_to_tensorboard(self, trajectory_list: List[EvaluationRun]):
        """
        Adds validation metrics to tensorboard - gets called by the trainingEngine

        Parameters
        ----------
        trajectory_list
            list of optimal trajectories for evaluation
        """
        samples_features = self.get_feature_tensor_from_list(trajectory_list)
        (
            average_lane_goal_reached,
            average_velocity_goal_reached,
        ) = utils.calculate_average_goals_reached(trajectory_list, 0.5)
        self.add_scalar_to_tensorboard(
            average_lane_goal_reached, "val/average_lane_goal_reached"
        )
        self.add_scalar_to_tensorboard(
            average_velocity_goal_reached, "val/average_velocity_goal_reached"
        )

        (
            average_lane_goal_reached_per_scenario,
            average_velocity_goal_reached_per_scenario,
        ) = utils.calculate_average_goals_reached_per_scenario(trajectory_list, 0.5)
        for scenario_key in average_lane_goal_reached_per_scenario:
            self.add_scalar_to_tensorboard(
                average_lane_goal_reached_per_scenario[scenario_key],
                "val/average_lane_goal_reached_" + str(scenario_key),
            )
            self.add_scalar_to_tensorboard(
                average_velocity_goal_reached_per_scenario[scenario_key],
                "val/average_velocity_goal_reached_" + str(scenario_key),
            )

        current_parameter = self.parameters.eval(session=self.sess)
        summary = utils.get_val_metrics(
            samples_features,
            current_parameter,
            self.expert_features,
            self.feature_names,
        )
        self.train_writer.add_summary(summary, self.train_counter)

    def add_experiment_metadata(self, number_of_experts: int, number_of_samples: int):
        """
        Add metadata of the training to tensorboard

        Parameters
        ----------
        number_of_experts
            number of expert trajectories
        number_of_samples
            number of proposal samples
        """
        # Source: https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/text/text_demo.py
        meta_info = {}
        meta_info["Training name"] = self.training_name
        meta_info["Irl model"] = "LinearIRL"
        meta_info["Reward Model"] = self.reward_model.get_name()
        meta_info["Sampling Policy"] = self.sampling_policy.get_name()
        meta_info["q Scale"] = self.sampling_policy.get_q_scale()
        meta_info["Number of experts"] = number_of_experts
        meta_info["Number of q samples"] = number_of_samples
        meta_info["Episode length"] = self.episode_length
        meta_info["Learning Rate"] = self.learning_rate
        meta_info["experts info"] = self.experts_summary
        meta_info["Initial value range"] = self.initial_parameters_magnitute
        meta_info["Number of features"] = self.number_of_features
        summary_op = tf.summary.text("Meta_Data", tf.constant(str(meta_info)))
        summary = self.sess.run(summary_op)
        self.train_writer.add_summary(summary)

    def add_tensorboard_summaries(self):
        """
        Adds current parameter values and normed parameter values as scalars to tensorboard
        """
        parameter_len = self.parameters.shape[0]
        parameter_sum = tf.reduce_sum(tf.abs(self.parameters))
        for i in range(0, parameter_len):
            tf.summary.scalar(
                name="param/" + self.feature_names[i], tensor=self.parameters[i][0]
            )
            tf.summary.scalar(
                name="param_normed/" + self.feature_names[i],
                tensor=tf.abs(self.parameters[i][0]) / parameter_sum,
            )
