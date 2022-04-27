import os
from typing import List, Tuple, Optional

import numpy as np
import tensorflow.compat.v1 as tf

from proseco.inverse_reinforcement_learning.trajectory import (
    EvaluationRun,
    Trajectory,
    ScenarioInfo,
)
from . import utils
from .baseIrlModel import BaseIRLModel

from proseco.utility.ui import get_logger
from proseco.utility.io import save_data

logger = get_logger("ProSeCo IRL", create_handler=False)

tf.disable_v2_behavior()


class NonLinearIRLModel(BaseIRLModel):
    def __init__(
        self,
        reward_model,
        sampling_policy,
        learning_rate,
        number_of_steps,
        training_name,
        optimizer_type,
        dimension_hidden_layer,
        add_bias_term,
        use_mini_batches,
        mini_batch_size,
        initial_values_range,
        regularize_weight,
        regularize_first_hidden_layer,
        work_dir,
    ):

        self.add_bias_term = add_bias_term
        self.use_mini_batches = use_mini_batches
        self.number_of_mini_batch_updates = 5
        self.mini_batch_size = mini_batch_size
        self.initial_values_range = initial_values_range
        self.dimension_hidden_layer = dimension_hidden_layer
        if self.use_mini_batches:
            self.updates_inner_loop = 1
        else:
            self.updates_inner_loop = self.number_of_mini_batch_updates

        self.regularize_first_hidden_layer = regularize_first_hidden_layer
        self.regularize_weight = regularize_weight

        self.w1 = None
        self.w2 = None
        self.b1 = None
        self.sum_elements = None
        self.loglikeli = None
        self.sample_rewards = None
        self.grads = None
        self.grad_norm = None
        self.loss = None

        super(NonLinearIRLModel, self).__init__(
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
        self.train_counter = 0

    ###### Training Methods #########

    def build_model(self):
        """
        Build tensorflow model for maximum likelihood of a linear return function give MaxEnt model

        Input tensors:
            expert_features_ph -- placeholder for expert features
            sampled_features_ph -- placeholder for features of sampled trajectories
            likelihood_samples_ph -- placeholder for likelihood values (importance weight of sample)

        Variables:
            parameters -- tensorflow variable which gets updated by SGD

        Target tensor:
            loglikeli -- Loglikelihood (estimate) of expert trajectories and current reward parameters

        Optimizer:
            optimizer -- SGD optimizer which minmizes -1*loglikeli (maximizes loglikeli)
        """

        # input placeholder are created for the expert trajectories (featurized), the sampled trajectories and the likelihood for the samples
        self.expert_features_ph = tf.placeholder(
            tf.float32, shape=[None, self.episode_length, self.number_of_features]
        )
        self.sampled_features_ph = tf.placeholder(
            tf.float32, shape=[None, self.episode_length, self.number_of_features]
        )
        self.likelihoods_samples_ph = tf.placeholder(tf.float32, shape=[None, 1])

        # main variables for the reward neural net
        self.w1 = tf.Variable(
            tf.random.uniform(
                shape=(self.number_of_features, self.dimension_hidden_layer),
                minval=-self.initial_values_range,
                maxval=self.initial_values_range,
                dtype=tf.float32,
            ),
            name="W1",
        )
        self.w2 = tf.Variable(
            tf.random.uniform(
                shape=(self.dimension_hidden_layer, 1),
                minval=-self.initial_values_range,
                maxval=self.initial_values_range,
                dtype=tf.float32,
            ),
            name="W2",
        )

        # adds bias to the first hidden layer if true
        if self.add_bias_term:
            self.b1 = tf.Variable(
                tf.random.uniform(
                    shape=(1, self.dimension_hidden_layer),
                    minval=-self.initial_values_range,
                    maxval=self.initial_values_range,
                    dtype=tf.float32,
                ),
                name="b1",
            )
        else:
            self.b1 = tf.zeros(
                shape=(1, self.dimension_hidden_layer), name="b1", dtype=tf.float32
            )

        ## forward pass likelihood
        ## expert rewards are calculated
        hidden_experts = tf.tensordot(self.expert_features_ph, self.w1, axes=[[2], [0]])
        hidden_experts = hidden_experts + self.b1
        hidden_experts = tf.nn.relu(hidden_experts)
        expert_step_rewards = tf.tensordot(hidden_experts, self.w2, axes=[[2], [0]])
        expert_rewards = tf.math.reduce_sum(expert_step_rewards, axis=1)

        ## sample rewards are calculated
        hidden_samples = tf.tensordot(
            self.sampled_features_ph, self.w1, axes=[[2], [0]]
        )
        hidden_samples = hidden_samples + self.b1
        hidden_samples = tf.nn.relu(hidden_samples)
        samples_step_rewards = tf.tensordot(hidden_samples, self.w2, axes=[[2], [0]])
        self.sample_rewards = tf.math.reduce_sum(samples_step_rewards, axis=1)

        ## Importance sampling estimate of partition function is calculated
        if self.sampling_policy.do_drop_importance_weights():
            self.sum_elements = (
                tf.math.exp(self.sample_rewards)
                / self.sampling_policy.get_artificial_importance_weight()
            )
        else:
            self.sum_elements = tf.multiply(
                tf.math.reciprocal(self.likelihoods_samples_ph),
                tf.math.exp(self.sample_rewards),
            )

        ## finally loglikelihood is calculated
        self.loglikeli = tf.math.reduce_mean(expert_rewards) - tf.math.log(
            tf.math.reduce_mean(self.sum_elements)
        )

        ## optimizer gets chosen
        optimizer = self.create_optimizer()

        ## decided of l2 regularization is added to first hidden layer
        if self.regularize_first_hidden_layer:
            norm_first_layer = tf.norm(self.w1)
            self.loss = -1 * self.loglikeli + self.regularize_weight * norm_first_layer
        else:
            self.loss = -1 * self.loglikeli

        self.grads = optimizer.compute_gradients(self.loss)

        ### logging parts for tensorboard
        if self.add_bias_term:
            grad_1, _ = self.grads[0]
            grad_2, _ = self.grads[1]
            grad_3, _ = self.grads[2]
            self.grad_norm = tf.math.sqrt(
                tf.math.reduce_sum(tf.math.pow(grad_1, 2))
                + tf.math.reduce_sum(tf.math.pow(grad_2, 2))
                + tf.math.reduce_sum(tf.math.pow(grad_3, 2))
            )
        else:
            grad_1, _ = self.grads[0]
            grad_2, _ = self.grads[1]
            self.grad_norm = tf.math.sqrt(
                tf.math.reduce_sum(tf.math.pow(grad_1, 2))
                + tf.math.reduce_sum(tf.math.pow(grad_2, 2))
            )
        tf.summary.scalar(name="train/grad_norm", tensor=self.grad_norm)

        # minimizing op
        self.step = optimizer.apply_gradients(self.grads)

        self.init_op = tf.variables_initializer(tf.global_variables())

        # tf sessions is created
        self.initialize_tf_session()
        self.initial_parameters = self.eval_current_weights()

    def make_update(self, trajectory_list: List[EvaluationRun]) -> Tuple[dict, dict]:
        """
        Main method which gets called by trainingEngine when stack is full

        Procedure:
            1. retrieves feature counts from trajectory list
            2. makes parameters update
            3. adjusts noise parameters (calling sampling_policy)
            4. creates new parameter and noise messages and returns them


        Parameters
        ----------
        trajectory_list
            list of trajectory message coming from the current stack

        Returns
        -------
        CostParam,NoiseParam
            new cost and noise message for service in trainingEngine after parameter update
        """
        self.assert_validity(trajectory_list)
        (
            samples_features,
            likelihood_tensor,
            average_selection_likelis,
            meta_info_trajectories,
        ) = self.get_trajectory_tensors_from_list(trajectory_list)

        new_parameters = self.make_parameters_update(
            samples_features, likelihood_tensor
        )

        return self.create_parameter_messages(new_parameters, expert=False)

    def make_parameters_update(
        self, samples_tensor: np.ndarray, likelihood_tensor: np.ndarray
    ) -> np.ndarray:
        """
        Main gradient update method - gets a sample feature count with likelis and makes a gradient descent update on the neg. loglikeli

        Parameters
        ----------
        samples_tensor
            numpy array with shape [number_of_trajectories,number_of_features] containing all featurecounts for the samples trajectories
        likelihood_tensor
            numpy array with shape [number_of_trajectories,1] containing the likelihoods of every trajectory under the current importance sampling policy

        Returns
        -------
        np.ndarray
            updated cost parameters after training step
        """

        logger.info(f"Update step: {self.train_counter}/{self.number_of_steps-1}")

        parameters_old = self.eval_current_weights()

        ### Loggings of the different parts of the sampled trajectories
        logger.debug("Parameters before update:")
        logger.debug(parameters_old)
        logger.debug("Likelis: ")
        logger.debug(likelihood_tensor)
        logger.debug("Shape expert features:")
        logger.debug(np.shape(self.expert_features))
        logger.debug("Shape sampled features:")
        logger.debug(np.shape(samples_tensor))
        logger.debug("Shape likelihood tensor:")
        logger.debug(np.shape(likelihood_tensor))
        train_summary = self.get_train_metrics(samples_tensor)

        ### Optimization step is performed (normal case is without mini batch and just one update)
        for _ in range(self.updates_inner_loop):
            if self.use_mini_batches:
                random_q_sample_index = np.random.choice(
                    np.shape(samples_tensor)[0], self.mini_batch_size, replace=False
                )
                random_expert_index = np.random.choice(
                    np.shape(self.expert_features)[0],
                    self.mini_batch_size,
                    replace=False,
                )
                q_samples_mini_batch = samples_tensor[random_q_sample_index, :, :]
                likelihoods_mini_batch = likelihood_tensor[random_q_sample_index, :]
                expert_mini_batch = self.expert_features[random_expert_index, :, :]
                (
                    merge_summary,
                    _,
                    gradient,
                    gradient_norm,
                    sum_elem,
                ) = self.run_update_session(
                    expert_mini_batch, q_samples_mini_batch, likelihoods_mini_batch
                )
            else:
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
        parameter_new = self.eval_current_weights()
        logger.debug(f"Parameters after update: {parameter_new}")

        if self.train_counter == 1:
            self.add_experiment_metadata(
                self.expert_features.shape[0], samples_tensor.shape[0]
            )

        ### add logs to tensorboard
        self.add_scalar_to_tensorboard(
            np.max(weight_of_sum_elem), "max_proposal/weight"
        )
        self.train_writer.add_summary(train_summary, self.train_counter)
        self.train_writer.add_summary(merge_summary, self.train_counter)
        ### Outputs end ###
        self.train_counter += 1

        return parameter_new

    def save_weights(self, save_path):
        w1_array = self.w1.eval(session=self.sess)
        w2_array = self.w2.eval(session=self.sess)
        b1_array = self.b1.eval(session=self.sess)
        # save_data(w1_array, os.path.join(save_path, "w1.json"), sort=False)
        # save_data(w2_array, os.path.join(save_path, "w2.json"), sort=False)
        # save_data(b1_array, os.path.join(save_path, "b1.json"), sort=False)
        np.savetxt(os.path.join(save_path, "w1.txt"), w1_array)
        np.savetxt(os.path.join(save_path, "w2.txt"), w2_array)
        np.savetxt(os.path.join(save_path, "b1.txt"), b1_array)

    def calculate_reward(self, feature_vec) -> float:
        """
        calculates the reward for a given input vec - forward pass through the neural net

        Parameters
        ----------
        feature_vec : np.array
            input feature vec to the reward neural net

        Returns
        -------
        float
            output value of the reward neural net
        """
        output = self.sess.run(
            [self.sample_rewards], feed_dict={self.sampled_features_ph: feature_vec}
        )
        return output[0]

    def eval_current_weights(self) -> dict:
        """
        Gets the current weights of the neural net from the tensorflow graph

        Returns
        -------
        dict
            dict with the weights of the neural net
        """
        parameters = {}
        parameters["w1"] = self.w1.eval(session=self.sess)
        parameters["w2"] = self.w2.eval(session=self.sess)
        parameters["b1"] = self.b1.eval(session=self.sess)
        assert not np.isnan(parameters["w1"]).any()
        assert not np.isnan(parameters["w2"]).any()
        assert not np.isnan(parameters["b1"]).any()
        logger.debug(parameters)
        return parameters

    ##### Trajectory preprocessing ########

    def extract_features_from_trajectory(
        self, trajectory_message: Trajectory, scenarioInfo: ScenarioInfo
    ) -> np.ndarray:
        """
        Extracts the features from one trajectory by iterating over all time steps and all agents

        Parameters
        ----------
        trajectory_message
            one sampled trajectory
        scenarioInfo
            info about the scenario.

        Returns
        -------
        np.array
            features of the complete trajectory
        """

        # initialize new feature vec which serves as an input to the neural net
        feature_vec = np.zeros(
            (self.episode_length, self.number_of_features), dtype=np.float32
        )
        # for some reward function the features of the previous state are also used
        previous_state_action = trajectory_message.initialState
        counter = 0

        # iterate over all state action pairs in the message and call feature_to_vec function of the reward model
        for state_action_pair in trajectory_message.trajectory:
            prev_agent_vec = previous_state_action
            agent_vec = state_action_pair

            feature_vec[counter, :] = self.reward_model.features_to_vec(
                agent_vec.features, prev_agent_vec.features, scenarioInfo
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
        # iterates over all single agent trajectory messages and creates the feature tensors that are inputs to the reward neural net
        for trajectory_message, scenarioInfo in self.split_trajectory_list(
            trajectory_list
        ):
            feature_vec = self.extract_features_from_trajectory(
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

        ## iterates over all single agent trajectory messages and creates the feature tensors that are inputs to the reward neural net + calculates the likelihood of the sample
        for trajectory_message, scenarioInfo in self.split_trajectory_list(
            trajectory_list
        ):
            feature_vec = self.extract_features_from_trajectory(
                trajectory_message, scenarioInfo
            )
            (
                likelihood,
                average_selection_likelihood,
            ) = self.extract_trajectory_likelihood(trajectory_message)
            samples.append(np.expand_dims(feature_vec, axis=0))
            likelis.append(np.expand_dims(likelihood, axis=0))
            average_selection_likelis.append(average_selection_likelihood)

            # Concatenate all feature vec for all num_samples trajectory to one np array
        samples_tensor = np.concatenate(samples, axis=0)
        likelihood_tensor = np.concatenate(likelis, axis=0)
        return samples_tensor, likelihood_tensor, average_selection_likelis, None

    ### Tensorboard # Logging methods ######

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

        summary = self.get_val_metrics(samples_features)
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
        meta_info["Irl Model"] = "NonlinearIRL"
        meta_info["Reward Model"] = self.reward_model.get_name()
        meta_info["Sampling Policy"] = self.sampling_policy.get_name()
        meta_info["q Scale"] = self.sampling_policy.get_q_scale()
        meta_info["Number of experts"] = number_of_experts
        meta_info["Number of q samples"] = number_of_samples
        meta_info["Episode length"] = self.episode_length
        meta_info["Learning Rate"] = self.learning_rate
        meta_info["experts info"] = self.experts_summary
        meta_info["Initial value range"] = self.initial_values_range
        meta_info["Number of hidden neurons"] = self.dimension_hidden_layer
        meta_info["Add bias"] = self.add_bias_term
        meta_info["Use minibatches"] = self.use_mini_batches
        meta_info["Minibatch size"] = self.mini_batch_size
        meta_info["do regularization"] = self.regularize_first_hidden_layer
        meta_info["regularizer weight"] = self.regularize_weight
        summary_op = tf.summary.text("Meta_Data", tf.constant(str(meta_info)))
        summary = self.sess.run(summary_op)
        self.train_writer.add_summary(summary)

    def retrieve_feature_metrics(
        self, samples_tensor: np.ndarray
    ) -> Tuple[float, float]:
        """
        calculates the number of collisions and invalids in a tensor

        Parameters
        ----------
        samples_tensor np.array - feature tensor of multiple trajectories

        Returns
        -------
        float
            number of collisions
        float
            number of invalids
        """
        collision_index = 3
        invalid_state_index = 4
        invalid_action_index = 5
        collision_count = 0.0
        invalid_count = 0.0
        invalid_action_count = 0.0
        number_of_trajectories = samples_tensor.shape[0]
        for trajectory_index in range(number_of_trajectories):
            if (
                np.count_nonzero(samples_tensor[trajectory_index, :, collision_index])
                > 0
            ):
                assert (
                    np.count_nonzero(
                        samples_tensor[trajectory_index, :, collision_index]
                    )
                    == 1
                )
                collision_count += 1.0
            if (
                np.count_nonzero(
                    samples_tensor[trajectory_index, :, invalid_state_index]
                )
                > 0
            ):
                assert (
                    np.count_nonzero(
                        samples_tensor[trajectory_index, :, invalid_state_index]
                    )
                    == 1
                )
                invalid_count += 1.0
            if (
                np.count_nonzero(
                    samples_tensor[trajectory_index, :, invalid_action_index]
                )
                > 0
            ):
                invalid_action_count += 1.0

        collision_amount = collision_count / float(number_of_trajectories)
        invalid_amount = invalid_count / float(number_of_trajectories)
        invalid_action_amount = invalid_action_count / float(number_of_trajectories)
        logger.debug("Collision amount: " + str(collision_amount * 100) + "%")
        logger.debug("Invalid amount: " + str(invalid_amount * 100) + "%")
        logger.debug("Invalid action amount: " + str(invalid_action_amount * 100) + "%")
        return collision_amount, invalid_amount

    def get_train_metrics(self, samples_tensor):
        """
        Calculates the metrics (reward,reward diff,collision,...) for training samples

        Parameters
        ----------
        samples_tensor : np.array
            feature tensor of proposal trajectories
        """
        collision_amount, invalid_amount = self.retrieve_feature_metrics(samples_tensor)
        sample_rewards = self.calculate_reward(samples_tensor)
        logger.debug("Sample rewards: ")
        logger.debug(sample_rewards)
        expert_rewards = self.calculate_reward(self.expert_features)
        logger.debug("Expert rewards: ")
        logger.debug(expert_rewards)
        mean_samples_reward = np.mean(sample_rewards)
        mean_experts_reward = np.mean(expert_rewards)
        reward_summary = tf.Summary()
        reward_summary.value.add(
            tag="train/Samples_avg_reward", simple_value=mean_samples_reward
        )
        reward_summary.value.add(
            tag="train/Experts_avg_reward", simple_value=mean_experts_reward
        )
        reward_summary.value.add(
            tag="train/avg_reward_diff",
            simple_value=mean_samples_reward - mean_experts_reward,
        )
        reward_summary.value.add(
            tag="train/avg_reward_diff_percent",
            simple_value=(mean_samples_reward - mean_experts_reward)
            / mean_experts_reward,
        )
        reward_summary.value.add(
            tag="train/collision_amount", simple_value=collision_amount
        )
        reward_summary.value.add(
            tag="train/invalid_amount", simple_value=invalid_amount
        )
        return reward_summary

    def get_val_metrics(self, samples_tensor):
        """
        Calculates the metrics (reward,reward diff,collision,...) for validation samples

        Parameters
        ----------
        samples_tensor : np.array
            feature tensor of sampled optimal trajectories
        """
        collision_amount, invalid_amount = self.retrieve_feature_metrics(samples_tensor)
        sample_rewards = self.calculate_reward(samples_tensor)
        expert_rewards = self.calculate_reward(self.expert_features)
        mean_samples_reward = np.mean(sample_rewards)
        mean_experts_reward = np.mean(expert_rewards)
        reward_summary = tf.Summary()
        reward_summary.value.add(
            tag="val/Samples_avg_reward", simple_value=mean_samples_reward
        )
        reward_summary.value.add(
            tag="val/Experts_avg_reward", simple_value=mean_experts_reward
        )
        reward_summary.value.add(
            tag="val/avg_reward_diff",
            simple_value=mean_samples_reward - mean_experts_reward,
        )
        reward_summary.value.add(
            tag="val/avg_reward_diff_percent",
            simple_value=(mean_samples_reward - mean_experts_reward)
            / mean_experts_reward,
        )
        reward_summary.value.add(
            tag="val/collision_amount", simple_value=collision_amount
        )
        reward_summary.value.add(tag="val/invalid_amount", simple_value=invalid_amount)
        return reward_summary
