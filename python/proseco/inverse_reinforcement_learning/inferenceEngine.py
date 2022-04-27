import os
import pickle
import numpy as np
from typing import Dict, List, Any


from proseco.inverse_reinforcement_learning.irl_models import utils
from proseco.inverse_reinforcement_learning.trainingEngine import TrainingEngine
from proseco.inverse_reinforcement_learning.modelBuilder import IrlConfig
from proseco.inverse_reinforcement_learning.irl_models.baseIrlModel import BaseIRLModel
from proseco.inverse_reinforcement_learning.trajectory import EvaluationRun
from proseco.inverse_reinforcement_learning.utils.evaluator import Evaluator
from proseco.utility.io import get_user_temp_dir, load_data, save_data
from proseco.utility.ui import let_user_select_subdirectory, get_logger

logger = get_logger("ProSeCo IRL", create_handler=False)


class InferenceEngine(TrainingEngine):
    def __init__(self, irl_model: BaseIRLModel, irl_config: IrlConfig):

        self.training_directory = let_user_select_subdirectory(
            str(get_user_temp_dir() / "irl/training")
        )
        # path to subdir where the parameters lie - for a linear model parameters.txt is searched in this dir and for the nonlinear reward w1.txt, w2.txt and b1.txt
        self.parameter_path = let_user_select_subdirectory(
            self.training_directory / "trajectories"
        )
        # Setable variables for inference

        # path to subdir where the expert pickle files lay (only needs to be set if compare with experts is true)
        self.expert_pickle_folder = str(
            get_user_temp_dir() / f"{irl_config.experts_path}/pickle_files"
        )

        # output folder where the metric summaries and plots should be put @todo make the scenario name a parameter
        self.output_path = irl_model.experiment_path
        os.makedirs(self.output_path, exist_ok=True)

        # name of the inference run @todo make the type of inference a parameter
        self.inference_name = f"{irl_config.training_name}"

        self.T = irl_config.episode_length

        super().__init__(irl_model, irl_config)

    def initialize(self):
        """
        Needs to be called at the beginning of the inference. Loads reward parameters and spins up the necessary ros components to communicate with the MCTS nodes
        """
        self.load_parameters()

    def load_parameters(self):
        """
        Loads learned reward parameters from .txt files -> parameters.txt for linear reward and w1.txt, w2.txt and b1.txt (which is always the zero vector right now )
        """
        noise_message = {
            "mean_y": [0],
            "sigma_y": [1],
            "mean_vx": [0],
            "sigma_vx": [1],
        }
        if self.irl_config.linear_reward:
            self.irl_model.reward_model.set_T(self.T)
            # parameters = load_data(self.parameter_path / "parameters.json")
            parameters = list(
                load_data(self.parameter_path / "parameters.json").values()
            )
            # parameters = [[x] for x in parameters]
            # parameters = [value for parameter in parameters for value in parameter]
            # parameters = np.loadtxt(os.path.join(self.parameter_path, "parameters.txt"))
            self.set_new_parameters(
                self.irl_model.reward_model.create_cost_message(parameters),
                noise_message,
                is_expert=False,
            )
        else:
            self.irl_model.reward_model.set_T(self.T)
            params = {}
            params["w1"] = np.loadtxt(os.path.join(self.parameter_path, "w1.txt"))

            params["w2"] = np.expand_dims(
                np.loadtxt(os.path.join(self.parameter_path, "w2.txt")), axis=1
            )

            params["b1"] = np.loadtxt(os.path.join(self.parameter_path, "b1.txt"))
            # params = [[x] for x in params]
            self.set_new_parameters(
                self.irl_model.reward_model.create_cost_message(params),
                noise_message,
                is_expert=False,
            )

    def infer(self):
        """
        Main inference loop
        """

        self.initialize()
        logger.info("Collecting trajectories")
        trajectory_list = self.collect_trajectories()
        logger.info("Starting evaluation")
        self.summarize(trajectory_list)

    def load_expert_pickles(self) -> List[EvaluationRun]:
        """Loads the expert trajectories from the pickle folder
        Returns
        -------
        List[EvaluationRun]
            The expert trajectories.
        """
        trajectories = []
        for scenario in self.irl_config.scenarios:
            file_names = os.listdir(self.expert_pickle_folder + "/" + scenario)
            for file_name in file_names:
                trajectory = pickle.load(
                    open(
                        os.path.join(self.expert_pickle_folder, scenario, file_name),
                        "rb",
                    )
                )
                trajectories.append(trajectory)
            trajectories = utils.filter_trajectory_list(trajectories)
        return trajectories

    def filter_trajectories_by_scenario(
        self, trajectories: List[EvaluationRun], scenario: str
    ) -> List[EvaluationRun]:
        """Filters the trajectories by scenario name

        Parameters
        ----------
        trajectories : List[EvaluationRun]
            List of evaluation runs (trajectories).
        scenario : str
            Name of the scenario.

        Returns
        -------
        List[EvaluationRun]
            The filtered list of evaluation runs (trajectories).
        """
        filtered_trajectories = []
        for trajectory in trajectories:
            if trajectory.scenarioInfo.scenarioName.lower() == scenario.lower():
                filtered_trajectories.append(trajectory)
        return filtered_trajectories

    def summarize_features(
        self,
        scenario: str,
        amount_lane_goal,
        amount_velocity_goal,
        amount_collisions,
        amount_invalids,
        amount_collisions_multi_agent,
        amount_invalids_multi_agent,
        amount_failed_multi_agent,
    ) -> Dict[str, Any]:
        """Creates a summary of some of the features of the trajectories

        Parameters
        ----------
        scenario : str
            The name of the scenario.
        amount_lane_goal : [type]
            The fraction of trajectories that reached the lane goal.
        amount_velocity_goal : [type]
            The fraction of trajectories that reached the velocity goal.
        amount_collisions : [type]
            The fraction of trajectories that collided with other vehicles.
        amount_invalids : [type]
            The fraction of trajectories that were invalid.
        amount_collisions_multi_agent : [type]
            The fraction of trajectories that collided with other vehicles (multi agent).
        amount_invalids_multi_agent : [type]
            The fraction of trajectories that were invalid (multi agent).
        amount_failed_multi_agent : [type]
            The fraction of trajectories that failed (multi agent).

        Returns
        -------
        Dict[str, Any]
            The summary of the features.
        """
        logger.debug(f"Lane goal reached: {amount_lane_goal * 100}%")
        logger.debug(f"Velocity goal reached: {amount_velocity_goal * 100}%")
        logger.debug(f"Collisions: {amount_collisions * 100}%")
        logger.debug(f"Invalids: {amount_invalids * 100}%")
        logger.debug(f"Multi-agent Collisions : {amount_collisions_multi_agent * 100}%")
        logger.debug(f"Multi-agent Invalids : {amount_invalids_multi_agent * 100}%")

        summary = {}
        summary["lane_goal"] = amount_lane_goal
        summary["velocity_goal"] = amount_velocity_goal
        summary["collisions"] = amount_collisions
        summary["invalids"] = amount_invalids
        summary["collisions_multi_agent"] = amount_collisions_multi_agent
        summary["invalids_multi_agent"] = amount_invalids_multi_agent
        summary["scenario"] = scenario
        summary["reward"] = self.irl_model.reward_model.get_name()
        summary["inference_name"] = self.inference_name
        summary["number_of_multi_agent_samples"] = self.irl_config.number_of_q_samples
        summary["amount_failed_multi_agent"] = amount_failed_multi_agent
        return summary

    def summarize_distances(
        self,
        scenario: str,
        avg_distance_to_nn_per_agent,
        std_distance_to_nn_per_agent,
    ) -> Dict[str, Any]:
        """Creates a summary of the distances from the optimal trajectories. And saves it to a file in the output_path.

        Parameters
        ----------
        scenario : str
            The name of the scenario.
        avg_distance_to_nn_per_agent : [type]
            The average distance to the nearest neighbor per agent.
        std_distance_to_nn_per_agent : [type]
            The standard deviation of the distance to the nearest neighbor per agent.

        Returns
        -------
        Dict[str, Any]
            The summary of the distances.
        """
        summary = {}
        for agent_key in avg_distance_to_nn_per_agent:
            summary_agent = {}
            summary_agent["avg_distance_knn"] = avg_distance_to_nn_per_agent[agent_key]
            summary_agent["std_distance_knn"] = std_distance_to_nn_per_agent[agent_key]
            summary[agent_key] = summary_agent
        summary["scenario"] = scenario
        summary["reward"] = self.irl_model.reward_model.get_name()
        summary["number_of_multi_agent_samples"] = self.irl_config.number_of_q_samples
        summary["inference_name"] = self.inference_name
        return summary

    def summarize(self, trajectory_list: List[EvaluationRun]):
        """
        Main method called by the infer method when the stack is full

        Arguments:
            trajectory_list [msg.Trajectory] - list of sampled trajectory messages
        """
        experts_trajectory_list = self.load_expert_pickles()

        feature_summaries = []
        distance_summaries = []
        for scenario in self.irl_config.scenarios:
            trajectories = self.filter_trajectories_by_scenario(
                trajectory_list, scenario
            )
            experts_trajectories = self.filter_trajectories_by_scenario(
                experts_trajectory_list, scenario
            )
            (
                amount_lane_goal_reached,
                amount_velocity_goal_reached,
            ) = utils.calculate_average_goals_reached(trajectories, 0.5)
            (
                amount_collisions,
                amount_invalids,
            ) = utils.count_number_of_collisions_and_invalids(trajectories)
            (
                amount_collisions_multi_agent,
                amount_invalids_multi_agent,
                amount_failed_multi_agent,
            ) = utils.count_number_of_collisions_and_invalids_multi_agent(trajectories)
            feature_summary = self.summarize_features(
                scenario,
                amount_lane_goal_reached,
                amount_velocity_goal_reached,
                amount_collisions,
                amount_invalids,
                amount_collisions_multi_agent,
                amount_invalids_multi_agent,
                amount_failed_multi_agent,
            )
            feature_summaries.append(feature_summary)
            save_data(
                feature_summary, f"{self.output_path}/feature_summary_{scenario}.json"
            )
            save_data(
                self.features_summary_by_scenario(feature_summaries),
                f"{self.output_path}/features_summary.json",
            )
            if self.irl_config.compare_with_experts:
                evaluator = Evaluator(
                    self.output_path, f"plot_{scenario}.png", int(self.T)
                )
                (
                    _,
                    _,
                    _,
                    _,
                    average_distance_to_nearest_neighbors_per_agent,
                    std_distance_to_nearest_neighbors_per_agent,
                    _,
                ) = evaluator.create(experts_trajectories, trajectories, False, True)
                distance_summary = self.summarize_distances(
                    scenario,
                    average_distance_to_nearest_neighbors_per_agent,
                    std_distance_to_nearest_neighbors_per_agent,
                )
                save_data(
                    distance_summary,
                    f"{self.output_path}/distance_summary_{scenario}.json",
                )
                distance_summaries.append(distance_summary)
            else:
                evaluator = Evaluator(
                    self.output_path, f"plot_{scenario}.png", int(self.T)
                )
                evaluator.create([], trajectories, False, False)

        if self.irl_config.compare_with_experts:
            save_data(
                self.distance_summary_by_scenario(distance_summaries),
                f"{self.output_path}/distance_summary.json",
            )

    @staticmethod
    def distance_summary_by_scenario(
        distance_summaries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Calculates the mean and standard deviation of the distance to the nearest neighbor per scenario

        Parameters
        ----------
        distance_summaries : List[Dict[str,Any]]
            The distance summaries.

        Returns
        -------
        Dict[str, Any]
            The mean and standard deviation of the distance to the nearest neighbor per scenario.
        """
        distance_summaries_by_scenario = []
        for distance_summary in distance_summaries:
            scenario = distance_summary["scenario"]
            distances_knn = []
            for key in distance_summary:
                if key.isnumeric():
                    distances_knn.append(distance_summary[key]["avg_distance_knn"])
            distance_summaries_by_scenario.append(
                {
                    "scenario": scenario,
                    "mean": np.mean(distances_knn),
                    "std": np.std(distances_knn),
                }
            )
        return distance_summaries_by_scenario

    @staticmethod
    def features_summary_by_scenario(
        feature_summaries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Summarizes the feature summaries by scenario in a list of dictionaries.

        Parameters
        ----------
        feature_summaries : List[Dict[str, Any]]
            The feature summaries.

        Returns
        -------
        List[Dict[str, Any]]
            The list feature summaries by scenario.
        """
        feature_summaries_by_scenario = []
        for feature_summary in feature_summaries:
            scenario = feature_summary["scenario"]
            feature_summaries_by_scenario.append(
                {
                    "scenario": scenario,
                    "failures": feature_summary["amount_failed_multi_agent"],
                    "collisions": feature_summary["collisions_multi_agent"],
                    "invalids": feature_summary["invalids_multi_agent"],
                    "velocity_goal": feature_summary["velocity_goal"],
                    "lane_goal": feature_summary["lane_goal"],
                }
            )
        return feature_summaries_by_scenario
