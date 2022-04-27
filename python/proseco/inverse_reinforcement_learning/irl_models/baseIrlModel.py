import dataclasses
import glob
import io
import os
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import tensorflow.compat.v1 as tf

from proseco.inverse_reinforcement_learning.reward_models.linear_rewards.baseLinearReward import (
    BaseLinearReward,
)
from proseco.inverse_reinforcement_learning.reward_models.non_linear_rewards.baseNonLinearReward import (
    BaseNonLinearReward,
)
from proseco.inverse_reinforcement_learning.sampling_policies.baseSamplingPolicy import (
    BaseSamplingPolicy,
)
from proseco.inverse_reinforcement_learning.trajectory import (
    EvaluationRun,
    ScenarioInfo,
    Trajectory,
)
from . import utils

from proseco.utility.io import create_timestamp, save_data
from proseco.utility.ui import get_logger

logger = get_logger("ProSeCo IRL", create_handler=False)

tf.disable_v2_behavior()


class BaseIRLModel(ABC):
    def __init__(
        self,
        reward_model: Union[BaseLinearReward, BaseNonLinearReward],
        sampling_policy: BaseSamplingPolicy,
        optimizer_type,
        learning_rate,
        number_of_steps,
        training_name,
        work_dir,
    ):
        self.reward_model = reward_model
        self.sampling_policy = sampling_policy
        self.feature_names = self.reward_model.get_feature_names()
        self.expert_cost_params = self.reward_model.get_expert_parameters()
        self.number_of_features = self.reward_model.get_number_of_features()

        self.experiment_path = None
        self.trajectory_path = None
        self.tensorboard_base_path = None
        self.expert_trajectory_map = defaultdict(
            list
        )  # maps scenarios to expert trajectories
        self.expert_features_map: Dict[
            str, np.ndarray
        ] = dict()  # maps scenarios to expert features
        self.all_expert_features = None  # expert features for all scenarios
        self.expert_features = None  # expert features that are actually used for the gradient optimization step

        self.sess = None
        self.train_writer = None
        self.merge = None

        self.learning_rate = learning_rate
        self.number_of_steps = number_of_steps
        self.training_name = training_name
        self.episode_length = 13
        self.reward_model.set_T(float(self.episode_length))
        self.working_dir = work_dir

        self.filter_expert_trajectories = True

        assert int(self.reward_model.T) == self.episode_length

        self.optimizer_type = optimizer_type
        self.experts_summary = {}

        #### DONT CHANGE ##############################
        self.initial_parameters = None
        self.expert_features_ph = None
        self.sampled_features_ph = None
        self.likelihoods_samples_ph = None
        self.parameters = None
        self.loglikeli = None
        self.init_op = None
        self.step = None
        self.train_counter = 0

    def create_paths(self):
        """
        Creates one base folder for the whole experiment and inside a folder for trajectories that are saved while training
        Inside the base folder also the logs folder is placed where the tensorboard logs are written
        """
        experiment_name = (
            create_timestamp()
            + "_"
            + self.training_name
            + "-"
            + self.reward_model.get_name()
            + "-"
            + self.sampling_policy.get_name()
        )
        self.experiment_path = os.path.join(self.working_dir, experiment_name)
        logger.info("Creating experiment path: " + self.experiment_path)
        os.makedirs(self.experiment_path, exist_ok=True)
        self.trajectory_path = os.path.join(self.experiment_path, "trajectories")
        logger.info("Creating trajectory path: " + self.trajectory_path)
        os.makedirs(self.trajectory_path, exist_ok=True)

    def save_config(self, irl_config):
        """Saves the configuration of the IRL training in the experiment folder.

        Parameters
        ----------
        irl_config : IrlConfig
            The configuration of the IRL training.
        """
        save_data(
            dataclasses.asdict(irl_config),
            os.path.join(self.experiment_path, "config.json"),
        )

    def set_tensorboard_path(self):
        self.tensorboard_base_path = os.path.join(self.experiment_path, "logs")

    def get_experiment_path(self):
        return self.experiment_path

    def get_trajectory_path(self):
        return self.trajectory_path

    @abstractmethod
    def build_model(self):
        """
        Builds the tensorflow model.
        """
        pass

    def create_optimizer(self) -> tf.train.Optimizer:
        """Creates the optimizer to be used for the gradient ascent on the reward model

        Returns
        -------
        tf.train.Optimizer
            The optimizer to be used for the gradient ascent on the reward model.
        """
        if self.optimizer_type == utils.Optimizer.SGD:
            return tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer_type == utils.Optimizer.MOMENTUM:
            return tf.train.MomentumOptimizer(
                learning_rate=self.learning_rate, momentum=0.9
            )
        elif self.optimizer_type == utils.Optimizer.ADAM:
            return tf.train.AdamOptimizer(learning_rate=self.learning_rate)

    def set_expert_trajectories(self, trajectory_list: List[EvaluationRun]):
        """
        Set a list of trajectory messages (either created or read from the pickle files) as expert trajectories
        - first the trajectories get filtered (collisions etc.)
        than the expert features tensor is created that is used to make the IRL updates

        Parameters
        ----------
        trajectory_list
            trajectory messages from the experts (either just generated or read from pickle files)
        """
        if self.filter_expert_trajectories:
            logger.info("Filtering expert trajectories")
            trajectory_list = utils.filter_trajectory_list(trajectory_list)
            self.assert_validity(trajectory_list)
            assert len(trajectory_list) > 0, "No valid expert trajectories found"

        for trajectory in trajectory_list:
            self.expert_trajectory_map[trajectory.scenarioInfo.scenarioName].append(
                trajectory
            )

        # determine expert features for each scenario seperately and store them in a map
        for scenario, scenario_trajectory_list in self.expert_trajectory_map.items():
            self.expert_features_map[scenario] = self.get_feature_tensor_from_list(
                scenario_trajectory_list
            )

        # create a summary for ALL expert trajectories
        self.get_summaries_for_expert_trajectory_messages(trajectory_list)

        self.all_expert_features = np.concatenate(
            list(self.expert_features_map.values()), axis=0
        )
        # initialize expert_features with all_expert_features. If `irl_config.only_matching_scenarios` is true,
        # the method `set_current_scenario` updates the expert_features according to the scenario of the current training iteration.
        self.expert_features = self.all_expert_features

        logger.debug(f"Expert features: {self.expert_features}")

    def set_experts_for_scenario(self, scenario: str) -> None:
        """Adjusts the IRL model so that only expert trajectories for the specified scenario are considered during the training iteration.

        Parameters
        ----------
        scenario : str
            scenario name
        """
        self.expert_features = self.expert_features_map.get(scenario.upper())

    def set_all_experts(self) -> None:
        """Adjusts the IRL model so that all expert trajectories are considered."""
        self.expert_features = self.all_expert_features

    def compare_to_episode_length(self, length: int) -> None:
        if length < self.episode_length:
            logger.debug(
                f"Trajectory shorter than episode length: {length} < {self.episode_length}"
            )
        else:
            logger.debug(
                "Trajectory at leas as long as episode length: {length} >= {self.episode_length}"
            )

    def get_summaries_for_expert_trajectory_messages(
        self, expert_trajectory_messages: List[EvaluationRun]
    ):
        """
        Calculates summary statistics for the expert trajectory_messages

        Parameters
        ----------
        expert_trajectory_messages
            trajectory messages from the experts (either just generated or read from pickle files)
        """
        initial_statistics = utils.calculate_average_initial_values_of_trajectories(
            expert_trajectory_messages
        )
        (
            average_lane_goal_reached,
            average_velocity_goal_reached,
        ) = utils.calculate_average_goals_reached(expert_trajectory_messages, 0.5)
        self.experts_summary["initial_statistics"] = initial_statistics
        self.experts_summary["average_lane_goal_reached"] = average_lane_goal_reached
        self.experts_summary[
            "average_velocity_goal_reached"
        ] = average_velocity_goal_reached
        logger.debug(f"Experts summary: {self.experts_summary}")

    @staticmethod
    def split_trajectory_list(
        trajectory_list: List[EvaluationRun],
    ) -> List[Tuple[Trajectory, ScenarioInfo]]:
        """Split the multi agent trajectories into single agent trajectories.

        Parameters
        ----------
        trajectory_list : List[EvaluationRun]
            list of evaluation runs that contain multi agent trajectories.

        Returns
        -------
        List[Tuple[Trajectory, ScenarioInfo]]
            list of tuples that contain a single agent trajectory and the corresponding scenario info.
        """
        flattened_trajectory_list = [
            (agent_trajectory, eval_run.scenarioInfo)
            for eval_run in trajectory_list
            for agent_trajectory in eval_run.trajectories
        ]
        logger.info(
            f"Extracting {len(flattened_trajectory_list)} single agent trajectories from scenarios."
        )
        return flattened_trajectory_list

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    def load_expert_trajectories(self):
        """
        Load expert feature counts from file
        """
        logger.info("Loading expert trajectories")

        pickle_base_path = self.reward_model.get_expert_pickle_folder()
        trajectory_messages = []
        # loads all pickle files from folder and adds the message to the messages list
        for file_name in glob.glob(
            pickle_base_path + "/**/expert_message_*.p", recursive=True
        ):
            with open(file_name, "rb") as f:
                trajectory_message = pickle.load(f)
                # trajectory_message.scenarioInfo.outputPath = re.sub('(/tmp/.{4,6}/)',f"/tmp/{getpass.getuser()}/",trajectory_message.scenarioInfo.outputPath)
                trajectory_messages.append(trajectory_message)

        self.set_expert_trajectories(trajectory_messages)

        # print("Check collisions for experts:")
        self.retrieve_feature_metrics(self.expert_features)

    def save_expert_trajectories_to_pickle(self, trajectory_list: List[EvaluationRun]):
        """
        Save a list of trajectory messages into pickle files (one file per message). Folder to save to is specfied
        in reward model

        Parameters
        ----------
        trajectory_list
            list of trajectory message coming from the current stack
        """
        counter = 0
        for trajectory_message in trajectory_list:
            save_dir = os.path.join(
                self.reward_model.get_expert_pickle_folder(),
                trajectory_message.scenarioInfo.scenarioName.lower(),
            )
            save_path = os.path.join(
                save_dir,
                "expert_message_" + str(counter + 1) + ".p",
            )
            os.makedirs(save_dir, exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(trajectory_message, f)
            counter += 1

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
        pass

    def get_initial_parameters(self):
        return self.create_parameter_messages(self.initial_parameters, expert=False)

    def get_expert_parameters(self):
        """
        Used for expert trajectory creation - give back a cost param message with the expert parameters
        """
        return self.create_parameter_messages(self.expert_cost_params, expert=True)

    def create_parameter_messages(self, cost_params: np.ndarray, expert: bool):
        """
        Set current cost and noise message which is retrieved to the slaves via the service

        Parameters
        ----------
        cost_params
            cost parameter vector
        expert
            true, if message should be created for experts.
        """
        if expert:
            cost_message = (
                self.reward_model.create_cost_message_expert(cost_params)
                if cost_params is not None
                else {}
            )
            noise_message = self.sampling_policy.create_expert_noise_message()
        else:
            cost_message = self.reward_model.create_cost_message(cost_params)
            noise_message = self.sampling_policy.create_noise_message()

        return cost_message, noise_message

    @abstractmethod
    def make_update(self, trajectory_list: List[EvaluationRun]):
        pass

    def run_update_session(
        self,
        experts_tensor: np.ndarray,
        samples_tensor: np.ndarray,
        likelihood_tensor: np.ndarray,
    ):
        return self.sess.run(
            [self.merge, self.step, self.grads, self.grad_norm, self.sum_elements],
            feed_dict={
                self.expert_features_ph: experts_tensor,
                self.sampled_features_ph: samples_tensor,
                self.likelihoods_samples_ph: likelihood_tensor,
            },
        )

    def done(self):
        return self.train_counter > self.number_of_steps - 1

    def get_final_selection_policy(self):
        """
        Retrieves name of the final selection policy that is given to the MCTS Node
        """
        final_selection_policy = self.sampling_policy.get_final_selection_policy()
        q_scale = self.sampling_policy.get_q_scale()
        return final_selection_policy, q_scale

    def close(self):
        self.close_tf_session()
        tf.reset_default_graph()

    def close_tf_session(self):
        self.sess.close()

    def extract_trajectory_likelihood(
        self, trajectory_message: Trajectory
    ) -> Tuple[np.ndarray, float]:
        """
        calculates the likelihood of one complete trajectory under the current sampling policy of the MCTS

        Parameters
        ----------
        trajectory_message
            one trajectory

        Returns
        -------
        np.ndarray
            1-d array with one value
        float
            average selection likelihood
        """
        (
            likelihood,
            average_selection_likelihood,
        ) = self.sampling_policy.calculate_trajectory_likelihood(
            trajectory_message, self.episode_length
        )
        return np.array([likelihood]), average_selection_likelihood

    def assert_validity(self, trajectory_list: List[EvaluationRun]):
        """
        checks the length of the episode and throws error if not

        Arguments:
            trajectory_list [] - list of sampled trajectories
        """
        for trajectory_message in trajectory_list:
            if utils.check_if_collision_or_invalid_appeared_in_trajectory_message(
                trajectory_message
            ):
                for agent in trajectory_message.trajectories:
                    assert len(agent.trajectory) == self.episode_length

    @abstractmethod
    def add_validation_to_tensorboard(self, trajectory_list: List[EvaluationRun]):
        """
        Adds validation metrics to tensorboard - gets called by the trainingEngine

        Parameters
        ----------
        trajectory_list
            list of optimal trajectories for evaluation
        """
        pass

    @abstractmethod
    def save_weights(self, current_trajectory_folder):
        pass

    def get_raw_expert_trajectory_messages(self, scenario_key):
        """
        Gives back the raw trajectory messages from the experts from the pickle files
        - this is used just for evaluation and plotting
        """
        return self.expert_trajectory_map[scenario_key]

    def get_episode_length(self):
        return self.episode_length

    def initialize_tf_session(self):
        """
        Initialize tf.sess and create writer file for tensorboard
        """
        self.sess = tf.Session()
        self.train_writer = tf.summary.FileWriter(
            self.tensorboard_base_path
            + "/"
            + create_timestamp()
            + "_"
            + self.training_name,
            self.sess.graph,
        )
        self.merge = tf.summary.merge_all()
        self.sess.run(self.init_op)

    def add_image_to_tensorboard(self, image_byte_buffer: io.BytesIO, name: str):
        """
        Adds an image to tensorboard - either trajectory plot or histogram - gets called from trainingEngine

        Parameters
        ----------
        image_byte_buffer
            buffer holding the image
        name
            name of the image
        """
        image = tf.image.decode_png(image_byte_buffer.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        # summary = self.sess.run(tf.summary.image(name, image,family=name))
        tf_image = tf.Summary.Image(
            encoded_image_string=image_byte_buffer.getvalue(), height=9, width=16
        )
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, image=tf_image)])

        self.train_writer.add_summary(summary, self.train_counter)

    def add_scalar_to_tensorboard(self, value: float, name: str):
        """
        Adds a scalar with name to the current tensorboard.

        Parameters
        ----------
        value
            scalar value
        name
            scalar name
        """
        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=value)
        self.train_writer.add_summary(summary, self.train_counter)
