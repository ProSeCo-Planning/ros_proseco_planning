import argparse
import itertools
import json
import os
import pickle
import random
import traceback

from typing import Iterable, List

import numpy as np
import tensorflow.compat.v1 as tf

from proseco.evaluator.head import Head
from proseco.inverse_reinforcement_learning.irl_models.baseIrlModel import BaseIRLModel
from proseco.inverse_reinforcement_learning.modelBuilder import IrlConfig
from proseco.inverse_reinforcement_learning.trajectory import EvaluationRun
from proseco.inverse_reinforcement_learning.utils.evaluator import Evaluator
from proseco.visualization.scenario_video_visualizer import ScenarioVideoVisualizer

from proseco.utility.io import get_number_digits
from proseco.utility.ui import get_logger

logger = get_logger("ProSeCo IRL", create_handler=False)

tf.disable_v2_behavior()


class TrainingEngine(Head):
    def __init__(self, irl_model: BaseIRLModel, irl_config: IrlConfig):
        """
        Initializes the master node.

        Parameters
        ----------
        irl_model
            an instance of an irl model.
        """

        #########################################
        # INITIALIZE EVALUATOR
        #########################################
        args = argparse.Namespace()
        args.no_dashboard = False
        args.debug = False
        args.no_summary = False
        args.config = "config.json"
        args.output = None
        args.address = None
        super().__init__(args=args)
        self.irl_model = irl_model
        self.irl_config = irl_config

        #########################################
        # OVERRIDE EVALUATOR BULK CONFIG
        #########################################

        self.bulk_config["options"] = [
            irl_config.options_experts
            if irl_config.create_experts
            else irl_config.options_irl
        ]

        # set the "scenarios" of the evaluator bulk_config. If `irl_config.only_matching_scenarios` is true,
        # the method `set_current_scenario` adjusts this parameter in each training iteration.
        self.bulk_config["scenarios"] = irl_config.scenarios
        # If `irl_config.only_matching_scenarios` is false, the q samples contain trajectories for several scenarios.
        # Hence, the number of runs for each scenario is a fraction of the number of q samples.
        # self.bulk_config["number_runs"] = (
        #     irl_config.number_of_q_samples
        #     if irl_config.only_matching_scenarios
        #     else int(irl_config.number_of_q_samples / len(irl_config.scenarios))
        # )

        options_alterations = {
            "compute_options": {
                "policy_options": {
                    "policy_enhancements": {"q_scale": [irl_config.q_scale]}
                },
                "random_seed": [
                    4209,
                    8928,
                    43,
                    2004,
                    4410,
                    269,
                    8551,
                    8914,
                    2817,
                    5723,
                    354,
                    8773,
                    9194,
                    9162,
                    7890,
                    3411,
                    3001,
                    9116,
                    5301,
                    1515,
                    4150,
                    8828,
                    468,
                    5430,
                    7641,
                    9117,
                    3564,
                    7413,
                    814,
                    7646,
                    7832,
                    8279,
                    9940,
                    9429,
                    9460,
                    5015,
                    7999,
                    6729,
                    8625,
                    4408,
                    69,
                    9888,
                    3324,
                    4382,
                    6940,
                    1753,
                    6412,
                    8648,
                    4021,
                    5748,
                    8440,
                    4288,
                    5327,
                    8786,
                    7304,
                    7668,
                    3703,
                    9846,
                    192,
                    7095,
                    6413,
                    1384,
                    1846,
                    9031,
                    1805,
                    5062,
                    9820,
                    8683,
                    1961,
                    9503,
                    1904,
                    6672,
                    689,
                    2621,
                    6421,
                ]
                # "random_seed": [
                #     0
                # ],  # guarantees that different runs can produce different trajectories
            },
        }
        self.bulk_config["options_alterations"] = options_alterations

        self.irl_model.sampling_policy.q_scale = irl_config.q_scale

        self.number_evaluations = self.determine_number_evaluations()

        #########################################
        # INITIALIZE SETTINGS
        #########################################

        # Expert Trajectories settings
        if irl_config.create_experts:
            self.save_expert_trajectories_to_pickle = True
            self.load_expert_trajectories_from_pickles = False
        else:
            self.save_expert_trajectories_to_pickle = False
            self.load_expert_trajectories_from_pickles = True

        self.collection_expert_flag = False

        # Validation settings
        if irl_config.create_experts:
            self.create_videos_for_experts = False
            self.number_of_videos_per_step = 16
            self.validate_first_iteration = False
        else:
            self.create_videos_for_experts = False
            self.number_of_videos_per_step = 16
            self.validate_first_iteration = True

        self.create_videos_in_validation = False
        self.create_plots_in_validation = False
        self.save_validation_trajectories = False
        self.add_images_to_tensorboard = True
        self.create_tsne_plots = False
        self.evaluate_trajectory_distances = True

        self.set_validation_frequency()
        self.current_trajectory_folder = ""
        self.current_trajectory_pickle_folder = ""

        # IO variables
        self.currentCostParams = None
        self.currentNoiseParams = None

    def set_validation_frequency(self) -> None:
        """Sets the validation frequency based on the current training iteration. This is a simple switch to lower the frequency after a certain amount of training iterations."""

        validation_frequency_first_part = 5
        validation_frequency_second_part = 10
        validation_increase_step = 50
        if self.irl_model.train_counter >= validation_increase_step:
            self.validation_frequency = validation_frequency_second_part
        else:
            self.validation_frequency = validation_frequency_first_part

    def is_validation_iteration(self) -> bool:
        """Returns true if the current iteration is a validation iteration."""

        return self.irl_model.train_counter % self.validation_frequency == 0

    def training_iteration(self, scenario_iterator: Iterable[str]):
        """The training iteration of the IRL algorithm.

        Parameters
        ----------
        scenario_iterator : Iterable[str]
            An infinite iterator over the scenarios that should be used for training.
        """
        logger.info("Training iteration")
        # only trajectories for a specific scenario are generated/considered in this iteration.
        if self.irl_config.only_matching_scenarios:
            # set the scenario for this iteration
            scenario = next(scenario_iterator)
            self.set_current_scenario(scenario)

        trajectory_list = self.collect_trajectories()

        # make a cost parameter update with the trajectories from the stack
        cost_model, action_noise = self.irl_model.make_update(trajectory_list)

        self.set_validation_frequency()

        # set the updated parameters as new parameters and creates a new cost id
        self.set_new_parameters(cost_model, action_noise, is_expert=False)

    def validation_iteration(self):
        """The validation iteration of the IRL algorithm."""
        logger.info("Validation iteration")

        self.create_validation_directories(self.irl_model.train_counter)

        if self.irl_config.only_matching_scenarios:
            # all scenarios are considered for the validation although `irl_config.only_matching_scenarios` is true
            self.set_all_scenarios()

        trajectory_list = self.collect_trajectories()

        # Make all validations: videos,metrics,plots
        self.irl_model.add_validation_to_tensorboard(trajectory_list)
        if self.create_videos_in_validation:
            self.create_videos()
        if self.create_plots_in_validation:
            self.create_plots(trajectory_list)

        # Save trajectories to pickle files
        if self.save_validation_trajectories:
            self.save_validation_trajectories_to_pickle_files(trajectory_list)

        # Save weights of irl model
        self.irl_model.save_weights(self.current_trajectory_folder)

    def train(self):
        """
        Main training loop
        """
        logger.info("Training started")

        # get experts either from files or generate new ones
        self.get_expert_trajectories()

        # set or reset start parameters as new cost function
        self.set_start_parameters()
        # iterator that cycles through the scenarios indefinitely
        scenario_iterator = itertools.cycle(self.irl_config.scenarios)

        # main trajectory collection loop
        while not self.irl_model.done():
            if self.is_validation_iteration():
                self.validation_iteration()
            self.training_iteration(scenario_iterator)

        self.validation_iteration()
        self.irl_model.close()

        logger.info("Training finished")

    def set_start_parameters(self):
        """
        Calls irl_model to get the initial cost and noise configuration and configures the service with them
        """
        (
            initial_cost_model,
            initial_action_noise,
        ) = self.irl_model.get_initial_parameters()
        self.set_new_parameters(
            initial_cost_model, initial_action_noise, is_expert=False
        )

    def get_expert_trajectories(self):
        """
        wrapper around methods for collecting or loading expert trajectories
        """
        if self.load_expert_trajectories_from_pickles:
            self.irl_model.load_expert_trajectories()
        else:
            self.collect_expert_trajectories()

    def collect_expert_trajectories(self):
        """
        Collect expert trajectories
        """
        self.collection_expert_flag = True
        (
            expert_cost_model,
            expert_action_noise,
        ) = self.irl_model.get_expert_parameters()
        self.set_new_parameters(expert_cost_model, expert_action_noise, is_expert=True)
        logger.info("Collecting expert features/trajectories")
        trajectory_list = self.collect_trajectories()
        logger.info("Collecting expert features/trajectories")
        self.irl_model.set_expert_trajectories(trajectory_list)
        if self.save_expert_trajectories_to_pickle:
            self.irl_model.save_expert_trajectories_to_pickle(trajectory_list)
        if self.create_videos_for_experts:
            logger.info("Create videos for experts")
            self.create_videos()
        self.collection_expert_flag = False

    def set_current_scenario(self, scenario: str) -> None:
        """Adjusts the evaluator bulk_config and the IRL model so that only proposal trajectories for the specified scenario are
        generated and only the expert trajectories for this scenario are considered.
        If `irl_config.only_matching_scenarios` is true, this method should be called in each training iteration.

        Parameters
        ----------
        scenario : str
            scenario name
        """
        # update the bulk_config for the evaluator
        self.bulk_config["scenarios"] = [scenario]
        # self.bulk_config["number_runs"] = self.irl_config.number_of_q_samples
        self.update_number_evaluations()
        # select the expert trajectories for the specified scenario
        self.irl_model.set_experts_for_scenario(scenario)

    def set_all_scenarios(self) -> None:
        """Adjusts the evaluator bulk_config and the IRL model so that trajectories for all scenarios are generated/considered."""
        # update the bulk_config for the evaluator
        self.bulk_config["scenarios"] = self.irl_config.scenarios
        # self.bulk_config["number_runs"] = self.irl_config.number_of_q_samples
        self.update_number_evaluations()
        # select all expert trajectories
        self.irl_model.set_all_experts()

    def update_number_evaluations(self) -> None:
        """Updates the evaluator member variable `number_evaluations` according to the current "bulk_config".

        Note
        ----
        If new alteration options are added to IRL config, the implementation must be adjusted.
        """
        self.number_evaluations = self.determine_number_evaluations()

    def collect_trajectories(self) -> List[EvaluationRun]:
        """Evaluates the current parameters using the evaluator.

        Returns
        -------
        List[EvaluationRun]
            A list of runs, where every run is a list of Trajectory-classes.
            Every Trajectory-class represents the trajectory of a single agent over the entire run.
        """

        # Start evaluator...
        results = self.start()

        extract_result = lambda res, prefix: next(
            (x for x in res if x[0].startswith(prefix))
        )[1]

        result = [
            (
                # All data for the IRL training:
                extract_result(res, "irl_trajectory"),
                # Trajectory annotated
                extract_result(res, "trajectory_annotated"),
                # Used to get the output path of every run:
                extract_result(res, "%PATH%"),
            )
            for res in results
        ]

        return [EvaluationRun.parse_eval_results(*x) for x in result]

    ####### Cost parameter methods ########

    def set_new_parameters(self, cost_model: dict, action_noise: dict, is_expert: bool):
        """
        Sets new cost model and action noise parameters.

        Parameters
        ----------
        cost_model
            The cost model to be used for the next training iteration.
        action_noise
            The action noise to be used for the next training iteration.
        is_expert
            True, if experts should be trained using the specified parameters, False otherwise
        """
        if all(key in cost_model for key in ("w1", "w2")):
            cost_model["w1"] = [cost_model["w1"].flatten().tolist()]
            cost_model["w2"] = [cost_model["w2"].flatten().tolist()]
        self.bulk_config["scenario_alterations"].update(
            {"agents": {"cost_model": cost_model}}
        )
        self.bulk_config["options_alterations"]["compute_options"].update(
            {"action_noise": action_noise}
        )
        self.bulk_config["evaluation_name"] = (
            "irl_experts"
            if is_expert
            else f"irl_{str(self.irl_model.train_counter).zfill(get_number_digits(self.irl_config.number_of_steps -1))}"
        )

    ############# Validation methods #############

    def create_validation_directories(self, training_iteration: int):
        """Creates the directories for the validation data. Used for the video creation.

        Parameters
        ----------
        training_iteration : int
            The current training iteration
        """
        self.current_trajectory_folder = os.path.join(
            self.irl_model.get_trajectory_path(), f"step{training_iteration:04}"
        )
        logger.debug(f"Creating validation folder: {self.current_trajectory_folder}")
        os.makedirs(self.current_trajectory_folder)
        if self.save_validation_trajectories:
            logger.debug("Creating pickle folder for validation trajectories")
            self.current_trajectory_pickle_folder = os.path.join(
                self.current_trajectory_folder, "pickle_folder"
            )
            os.makedirs(self.current_trajectory_pickle_folder)

    def save_validation_trajectories_to_pickle_files(
        self, trajectory_list: List[EvaluationRun]
    ):
        """
        Saves validation trajectories to a pickle file.

        Parameters
        ----------
        trajectory_list
            trajectories from the stack
        """
        counter = 1
        logger.debug("Saving trajectories to pickle files")
        for trajectory_message in trajectory_list:
            save_path = os.path.join(
                self.current_trajectory_pickle_folder,
                "trajectory_message_" + str(counter) + ".p",
            )
            pickle.dump(trajectory_message, open(save_path, "wb"))
            counter += 1

    def create_videos(self):
        """
        Create videos from specified folder - calls video_creator object for that purpose
        """
        print("-Create videos")
        folder = self.path
        subdirs = [
            directory
            for directory in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, directory))
        ]
        # get most recent directories
        subdirs.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
        subdirs_sampled = random.sample(
            subdirs[0 : int(self.irl_config.number_of_q_samples)],
            k=self.number_of_videos_per_step,
        )
        for directory in subdirs_sampled:
            try:
                with open(
                    os.path.join(folder, directory, "scenario_output.json"), "r"
                ) as f:
                    scenario = json.load(f)["name"].lower()
                video_creator = ScenarioVideoVisualizer(
                    os.path.join(folder, directory, "trajectory_annotated.json"),
                )
                video_creator.save(os.path.join(folder, directory, "video.mp4"))
            except Exception:
                traceback.print_exc()

    def create_plots(self, trajectory_list: List[EvaluationRun]):
        """
        Creates validation plots for all scenarios in the run - calls evaluator object to create trajectory plots and calculate trajectory distances

        Parameters
        ----------
        trajectory_list
            trajectories from the stack
        """

        all_distances_to_nearest_neighbors_all_scenarios = []
        # iterates over all scenarios
        for scenario_key in self.bulk_config["scenarios"]:
            ### Expert trajectory paths
            expert_trajectory_list = self.irl_model.get_raw_expert_trajectory_messages(
                scenario_key.upper()
            )
            # trajectories for this scenario
            scenario_trajectory_list = [
                trajectory
                for trajectory in trajectory_list
                if trajectory.scenarioInfo.scenarioName == scenario_key.upper()
            ]

            evaluator = Evaluator(
                self.current_trajectory_folder,
                "plot_" + str(scenario_key) + ".png",
                self.irl_model.get_episode_length(),
            )

            # Main creation of all validation plots/number with the Evaluator
            (
                main_fig_buffer,
                tsne_buffer,
                hist_buffer,
                average_distances,
                average_distance_to_nearest_neighbors_per_agent,
                std_distance_to_nearest_neighbors_per_agent,
                all_distances_to_nearest_neighbors_in_scenario,
            ) = evaluator.create(
                expert_trajectory_list,
                scenario_trajectory_list,
                self.create_tsne_plots,
                self.evaluate_trajectory_distances,
            )
            all_distances_to_nearest_neighbors_all_scenarios += (
                all_distances_to_nearest_neighbors_in_scenario
            )

            # Add trajectory plots to tensorbaord
            if self.add_images_to_tensorboard:
                self.irl_model.add_image_to_tensorboard(
                    main_fig_buffer, "plot_" + str(scenario_key)
                )

            # Add tsne plots to tensorboard
            if self.create_tsne_plots:
                self.irl_model.add_image_to_tensorboard(
                    tsne_buffer, "tsne_" + str(scenario_key)
                )

            # Add distance plots + metrics to tensorboard
            if self.evaluate_trajectory_distances:
                self.irl_model.add_image_to_tensorboard(
                    hist_buffer, "hist_" + str(scenario_key)
                )
                for agent_key in average_distances:
                    self.irl_model.add_scalar_to_tensorboard(
                        name=f"val/{scenario_key}_{agent_key}_avg_distance",
                        value=average_distances[agent_key],
                    )
                for agent_key in average_distance_to_nearest_neighbors_per_agent:
                    self.irl_model.add_scalar_to_tensorboard(
                        name=f"val/{scenario_key}_{agent_key}_avg_distance_nearest_neighbor",
                        value=average_distance_to_nearest_neighbors_per_agent[
                            agent_key
                        ],
                    )
                    self.irl_model.add_scalar_to_tensorboard(
                        name=f"val/{scenario_key}_{agent_key}_std_distance_nearest_neighbor",
                        value=std_distance_to_nearest_neighbors_per_agent[agent_key],
                    )

        # calc distance summaries for all trajectories over all scenarios
        if self.evaluate_trajectory_distances:
            average_distance = np.mean(
                np.array(all_distances_to_nearest_neighbors_all_scenarios)
            )
            std_distance = np.std(
                np.array(all_distances_to_nearest_neighbors_all_scenarios)
            )
            logger.debug(
                "Average distance to nearest neighbors of all agents in all scenarios:"
                + str(average_distance)
            )
            logger.debug(
                "Std of nearest neighbors of all agents in all scenarios:"
                + str(std_distance)
            )
            self.irl_model.add_scalar_to_tensorboard(
                name="val/avg_distance_nearest_neighbor", value=average_distance
            )
            self.irl_model.add_scalar_to_tensorboard(
                name="val/std_distance_nearest_neighbor", value=std_distance
            )
            self.irl_model.add_scalar_to_tensorboard(
                name="val/len_distances_nearest_neighbor",
                value=len(all_distances_to_nearest_neighbors_all_scenarios),
            )
