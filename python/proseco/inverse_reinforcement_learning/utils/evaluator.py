from typing import List, Optional, Tuple, Union, Any, Dict

import matplotlib
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from proseco.inverse_reinforcement_learning.trajectory import EvaluationRun

matplotlib.use("Cairo")

import os
import numpy as np
import matplotlib.pyplot as plt
from proseco.utility import import_proseco_data
from proseco.utility.visualize_proseco_data import TrajectoryVisualizer
import io
from sklearn.manifold import TSNE
from enum import Enum
import pickle
import getpass

from proseco.inverse_reinforcement_learning.irl_models import utils
from proseco.utility.ui import get_logger

logger = get_logger("ProSeCo IRL", create_handler=False)


class Distance(Enum):
    """
    Specifies the metric which is used to calculate the distance between two trajectories
    """

    EUCLIDEAN = 0
    EUCLIDEANSQUARED = 1
    MAXNORM = 2


class Evaluator:
    def __init__(
        self,
        save_folder,
        save_file_name,
        episode_length,
    ):
        self.save_folder = save_folder
        self.save_file_name = save_file_name
        self.figureWidth = 20.0
        self.dpiNumber = 80.0
        self.faceColor = "w"
        self.edgeColor = "k"
        self.imageFormat = 9.0 / 16.0
        self.single_agent = False
        if self.single_agent:
            self.imageFormat = self.imageFormat / 2.0
        self.m_textsize = 18
        self.steps_in_trajectory = 8 * episode_length + 1
        self.draw_agents = False
        self.expert_tag = "exp"
        self.irl_tag = "irl"
        self.trajectory_annotationSize = 15
        self.distance_type = Distance.EUCLIDEAN
        self.number_of_nearest_neighbors = 3

    def create_figure(self, identifier: Union[int, str, None] = None) -> Figure:
        """Creates a figure with the evaluator specific settings.

        Parameters
        ----------
        identifier : Union[int, str], optional
            The identifier of the figure, by default None

        Returns
        -------
        Figure
            The figure with the evaluator specific settings.
        """
        return plt.figure(
            num=identifier,
            figsize=(self.figureWidth, self.imageFormat * self.figureWidth),
            dpi=self.dpiNumber,
            facecolor=self.faceColor,
            edgecolor=self.edgeColor,
        )

    def create(
        self,
        expert_runs: Optional[List[EvaluationRun]],
        irl_runs: Optional[List[EvaluationRun]],
        create_tsne: bool,
        evaluate_distances: bool,
    ):
        """Main method of class - gets mutliple subfolders of the root folders containing the actual files of single trajectories (for experts and for irl trajectories)

        Arguments:
            expert_folders [string] -- list of subdirectories where the individual expert trajectory is in
            irl_folders [string] -- list of subdirectories where the individual irl trajectory is in

        Return:
            trajectory_buffer BytesIO -- in memory byte stream containing the trajectory plot - can be processed by tensorboard
            tsne_fig_buffer BytesIO -- in memory byte stream containing the tsne plot - can be processed by tensorboard

        """
        assert expert_runs or irl_runs, "Need either expert runs or irl runs"
        all_runs = (expert_runs or []) + (irl_runs or [])
        assert all(
            run.scenarioInfo.scenarioName == all_runs[0].scenarioInfo.scenarioName
            for run in all_runs
        )
        numberAgents = len(all_runs[0].trajectories)

        plt.xticks(fontsize=10)

        # initialize figures
        trajectory_fig = self.create_figure("Vehicle Movement 2D")
        trajectory_fig.tight_layout()
        trajectory_fig.subplots_adjust(hspace=0.5)

        tsne_fig = self.create_figure()
        hist_fig = self.create_figure()

        trajectory_axes = []
        tsne_axes = []
        hist_axes = []
        all_trajectories = {}
        average_distances = {}

        # create and store all axes
        for i in range(numberAgents):
            all_trajectories[f"agent{i}"] = []
            average_distances[f"agent{i}"] = 0.0
            trajectory_axes.append(trajectory_fig.add_subplot(numberAgents, 1, 1 + i))
            tsne_axes.append(tsne_fig.add_subplot(1, numberAgents, 1 + i))
            hist_axes.append(hist_fig.add_subplot(1, numberAgents, 1 + i))

        # add expert trajectories to trajectory plot
        if expert_runs:
            self.add_trajectories_to_plot(
                expert_runs, trajectory_fig, all_trajectories, expert=True
            )

        # add irl trajectories to trajectory plot
        if irl_runs:
            self.add_trajectories_to_plot(
                irl_runs, trajectory_fig, all_trajectories, expert=False
            )

        # if only expert or irl should be plotted tsne and distance calc is not possible
        if not expert_runs or not irl_runs:
            create_tsne = False
            evaluate_distances = False

        # Save trajectory plot in buffer for tensorboard and to file
        trajectory_buffer = io.BytesIO()
        trajectory_fig.savefig(trajectory_buffer, format="png")
        trajectory_fig.savefig(
            os.path.join(self.save_folder, self.save_file_name), format="png"
        )
        logger.debug("Saving trajectory plot")

        # Create tsne plot from distances
        tsne_fig_buffer = io.BytesIO()
        if create_tsne:
            for i in range(numberAgents):
                self.tsne_plot(tsne_axes[i], all_trajectories, i)
            logger.debug("Saving tsne plot")
            tsne_fig.savefig(tsne_fig_buffer, format="png")
            tsne_fig.savefig(
                os.path.join(self.save_folder, "tsne_" + self.save_file_name),
                format="png",
            )

        # Evaluate distances between expert and irl trajectories - calc distances of nearest neighbors, histograms from the distances
        hist_fig_buffer = io.BytesIO()
        average_distance_to_nearest_neighbors_per_agent = {}
        std_distance_to_nearest_neighbors_per_agent = {}
        all_distances_to_nearest_neighbors = []
        if evaluate_distances:
            for i in range(numberAgents):
                (
                    distances,
                    distances_per_irl_trajectory,
                ) = self.calculate_irl_exp_distances(all_trajectories, i)
                average_distances["agent" + str(i)] = np.mean(distances)
                self.histogram_plot(hist_axes[i], distances)
                agents_average_distance_to_nearest_neighbor = []
                for key in distances_per_irl_trajectory:
                    distances_per_irl_list = distances_per_irl_trajectory[key]
                    distances_per_irl_list.sort()
                    average_distance_to_nearest_neighbor = np.mean(
                        np.array(
                            distances_per_irl_list[: self.number_of_nearest_neighbors]
                        )
                    )
                    agents_average_distance_to_nearest_neighbor.append(
                        average_distance_to_nearest_neighbor
                    )
                    all_distances_to_nearest_neighbors.append(
                        average_distance_to_nearest_neighbor
                    )
                average_distance_to_nearest_neighbors_per_agent[str(i)] = np.mean(
                    np.array(agents_average_distance_to_nearest_neighbor)
                )
                std_distance_to_nearest_neighbors_per_agent[str(i)] = np.std(
                    np.array(agents_average_distance_to_nearest_neighbor)
                )
                logger.debug(
                    f"avg distance to nearest neighbors | agent {i}: {average_distance_to_nearest_neighbors_per_agent[str(i)]:.2f}"
                )
                logger.debug(
                    f"std distance to nearest neighbors | agent {i}: {std_distance_to_nearest_neighbors_per_agent[str(i)]:.2f}"
                )

            logger.debug(f"Saving nearest neighbors histogram")
            hist_fig.savefig(hist_fig_buffer, format="png")
            hist_fig.savefig(
                os.path.join(self.save_folder, "hist_" + self.save_file_name),
                format="png",
            )
        plt.close("all")
        return (
            trajectory_buffer,
            tsne_fig_buffer,
            hist_fig_buffer,
            average_distances,
            average_distance_to_nearest_neighbors_per_agent,
            std_distance_to_nearest_neighbors_per_agent,
            all_distances_to_nearest_neighbors,
        )

    def read_trajectory(
        self, evaluation_run: EvaluationRun
    ) -> Tuple[Dict[str, Any], np.ndarray, int, TrajectoryVisualizer, str]:
        """[summary]

        Parameters
        ----------
        evaluation_run : EvaluationRun
            [description]

        Returns
        -------
        Tuple[Dict[str, Any], np.ndarray, int, TrajectoryVisualizer, str]
            The trajectory, obstacles, number of agents, trajectory visualizer and scenario name.
        """

        trajectory_annotated = evaluation_run.trajectoryAnnotated
        scenarioName = evaluation_run.scenarioInfo.scenarioName
        numberAgents = len(trajectory_annotated["agents"])
        agentsMaxSteeringAngle = np.empty(shape=[numberAgents])
        agentsMaxAbsoluteAcceleration = np.empty(shape=[numberAgents])

        for i in range(numberAgents):
            agentsMaxAbsoluteAcceleration[i] = trajectory_annotated["agents"][i][
                "vehicle"
            ]["max_acceleration"]
            agentsMaxSteeringAngle[i] = import_proseco_data.convert_rad_to_deg(
                trajectory_annotated["agents"][i]["vehicle"]["max_steering_angle"]
            )
        agentsLimits = {
            "agentsMaxAbsoluteAcceleration": agentsMaxAbsoluteAcceleration,
            "agentsMaxSteeringAngle": agentsMaxSteeringAngle,
        }
        trajectory = import_proseco_data.get_driven_trajectory(trajectory_annotated)
        obstacles = import_proseco_data.get_static_obstacles(trajectory_annotated)
        agentsPredefined = []
        for i in range(numberAgents):
            isPredefined = trajectory_annotated["agents"][i]["is_predefined"]
            agentsPredefined.append(isPredefined)
        trajectory["agentsPredefined"] = agentsPredefined
        visualizer = TrajectoryVisualizer(
            trajectory["agentsPositionX"],
            agentsLimits,
            trajectory["numberLanes"],
            trajectory["laneWidth"],
            self.m_textsize,
            "english",
        )

        return trajectory, obstacles, numberAgents, visualizer, scenarioName

    def add_trajectories_to_plot(
        self,
        runs: List[EvaluationRun],
        figure: Figure,
        all_trajectories: List[Any],
        expert: bool,
    ) -> None:
        """Adds trajectories to the plot.

        Parameters
        ----------
        runs : List[EvaluationRun]
            The list of runs that contain the trajectories to be added to the plot.
        figure : Figure
            The figure to which the trajectories should be added.
        all_trajectories : List[Any]
            The list of tuples with x,y coordinates describing the trajectories.
        expert : bool
            Whether the trajectories are expert or IRL trajectories.
        """
        tag = self.expert_tag if expert else self.irl_tag
        for run in runs:
            try:
                (
                    trajectory,
                    obstacles,
                    numberAgents,
                    visualizer,
                    scenarioName,
                ) = self.read_trajectory(run)
                for i in range(numberAgents):
                    self.add_trajectory_to_plot(
                        figure,
                        figure.axes[i],
                        i,
                        trajectory,
                        obstacles,
                        numberAgents,
                        visualizer,
                        expert,
                    )
                    x, y = self.extract_x_y_points(
                        i, trajectory, numberAgents, self.steps_in_trajectory
                    )
                    all_trajectories["agent" + str(i)].append((tag, x, y))
            except Exception:
                logger.error(
                    f"Error in expert run {run.scenarioInfo.outputPath}", exc_info=True
                )

    def add_trajectory_to_plot(
        self,
        fig: Figure,
        ax: Axes,
        agentIndex: int,
        trajectory: dict,
        obstacles,
        numberAgents: int,
        visualizer: TrajectoryVisualizer,
        expert: bool,
    ):
        """Adds one trajectory to the figure, colors differentiate expert and IRL trajectories.

        Parameters
        ----------
        fig : Figure
            The figure to which the trajectory is added.
        ax : Axes
            The axes to which the trajectory is added.
        agentIndex : int
            The index of the agent.
        trajectory : dict
            The driven trajectory of the agent.
        obstacles : [type]
            The obstacles of the scenario.
        numberAgents : int
            The number of agents in the scenario.
        visualizer : TrajectoryVisualizer
            The trajectory visualizer object, for this single trajectory.
        expert : bool
            Whether the trajectory is an expert trajectory (differentiates the color).
        """

        ax = visualizer.drawTrajectory2DSingleAgent(
            fig, ax, agentIndex, trajectory, expert, False
        )
        ax = visualizer.drawStaticObstacles(ax, obstacles)
        if self.draw_agents:
            for i in range(numberAgents):
                index = 0
                ax, _ = visualizer.drawObject(
                    ax,
                    "vehicle",
                    trajectory["agentsPositionX"][index, i],
                    trajectory["agentsPositionY"][index, i],
                    trajectory["agentsHeading"][index, i],
                    trajectory["agentsVehicleLength"][index, i],
                    trajectory["agentsVehicleWidth"][index, i],
                )
                if numberAgents > 3 and i > 0:
                    continue
                ax.annotate(
                    "$g_" + str(i) + "$",
                    xy=(
                        trajectory["agentsPositionX"][index, i],
                        trajectory["agentsPositionY"][index, i],
                    ),
                    xytext=(
                        trajectory["agentsPositionX"][index, i],
                        trajectory["agentsPositionY"][index, i]
                        - trajectory["agentsVehicleWidth"][index, i],
                    ),
                    fontsize=self.trajectory_annotationSize,
                    bbox=dict(boxstyle="circle", fc="w", ec="k"),
                )
                index = -1
                ax, _ = visualizer.drawObject(
                    ax,
                    "vehicle",
                    trajectory["agentsPositionX"][index, i],
                    trajectory["agentsPositionY"][index, i],
                    trajectory["agentsHeading"][index, i],
                    trajectory["agentsVehicleLength"][index, i],
                    trajectory["agentsVehicleWidth"][index, i],
                )
                ax.annotate(
                    "$g_" + str(i) + "$",
                    xy=(
                        trajectory["agentsPositionX"][index, i],
                        trajectory["agentsPositionY"][index, i],
                    ),
                    xytext=(
                        trajectory["agentsPositionX"][index, i],
                        trajectory["agentsPositionY"][index, i]
                        - trajectory["agentsVehicleWidth"][index, i],
                    ),
                    fontsize=self.trajectory_annotationSize,
                    bbox=dict(boxstyle="circle", fc="w", ec="k"),
                )
        ax.set_xlim([-50.0, 180.0])

    def extract_x_y_points(
        self, agentIndex: int, drivenTrajectory, numberAgents, steps
    ):
        """Method for extracting the x,y curve from the drivenTrajectory dict -- trajectories that were in a terminal state gets enlargend

        Arguments:
            agentIndex -- index of Agent
            drivenTrajectory -- dict
            obstacles -- dict
            numberAgents -- number of agents
            steps -- int number of time steps in the trajectory -- smaller trajectories (that were in a terminal state) gets expanded to that length by staying in the last position for the difference of lenght and steps (as is clear for a terminal state)
        """
        time = drivenTrajectory["stage"]
        x_coordinates = []
        y_coordinates = []
        x_coordinates_old = drivenTrajectory["agentsPositionX"][:, agentIndex]
        y_coordinates_old = drivenTrajectory["agentsPositionY"][:, agentIndex]
        previous_stage_step = 0
        counter = 0
        for stage_step in time:
            if not stage_step == previous_stage_step:
                previous_stage_step = stage_step
                continue

            x_coordinates.append(x_coordinates_old[counter])
            y_coordinates.append(y_coordinates_old[counter])

            counter += 1

        length = len(x_coordinates)
        if length < steps:
            new_x_coordiante = np.empty((steps))
            new_x_coordiante.fill(x_coordinates[-1])
            new_x_coordiante[0:length] = x_coordinates
            new_y_coordiante = np.empty((steps))
            new_y_coordiante.fill(y_coordinates[-1])
            new_y_coordiante[0:length] = y_coordinates
            x_coordinates = new_x_coordiante
            y_coordinates = new_y_coordiante
        return x_coordinates, y_coordinates

    def calculate_distance_of_trajectories(self, trajectory1, trajectory2):
        """Method for calculating the distance between two trajectories - distance is approx integral (over t) of the squared norm between the two curves (x1(t)-x2(t))^2+(y1(t)-y2(t))^2

        Arguments:
            trajectory1 -- (tag,x,y) first trajecotry containing the tag if it is an expert trajectory and the arrays of x and y coordinates
            trajectory2 -- (tag,x,y) second trajecotry containing the tag if it is an expert trajectory and the arrays of x and y coordinates

        Return:
            distance -- float approx integral (over t) of the squared norm between the two curves (x1(t)-x2(t))^2+(y1(t)-y2(t))^2
        """
        _, x1, y1 = trajectory1
        _, x2, y2 = trajectory2
        length1 = np.shape(x1)[0]
        length2 = np.shape(x2)[0]

        assert length1 == length2
        cumulated_norm = 0.0
        norm_list = []
        for i in range(length1):
            if (
                self.distance_type == Distance.EUCLIDEAN
                or self.distance_type == Distance.MAXNORM
            ):
                norm = np.sqrt(
                    np.power(x1[i] - x2[i], 2.0) + np.power(y1[i] - y2[i], 2.0)
                )

            if self.distance_type == Distance.EUCLIDEANSQUARED:
                norm = np.power(x1[i] - x2[i], 2.0) + np.power(y1[i] - y2[i], 2.0)
            norm_list.append(norm)
            cumulated_norm += norm
        average_distance = cumulated_norm / float(length1)
        max_distance = max(norm_list)

        # print(max_distance)
        if self.distance_type == Distance.MAXNORM:
            return max_distance
        else:
            return average_distance

    def calculate_irl_exp_distances(self, all_trajectories, agentIndex):
        """
        Calculates all distances between experts and irl trajectories

        Arguments:
            all_trajectories dict - dictionary containing lists of trajectories for each agent (key = agent0 agent1...)
            agentIndex int - index of agent for which the distances to the experts should be calculated

        Return:
            [] - list of all distances between irl and experts
            dict - containing the distances to the experts for each single irl trajectories key= index of irl trajectory
        """
        trajectories = all_trajectories["agent" + str(agentIndex)]
        number_of_trajectories = len(trajectories)
        distances = []
        distances_per_irl_trajectory = {}
        irl_trajectory_counter = 0
        for i in range(number_of_trajectories):
            trajectory_type, _, _ = trajectories[i]
            if trajectory_type == self.irl_tag:
                distances_per_irl_trajectory[irl_trajectory_counter] = []
                for j in range(number_of_trajectories):
                    trajectory_type_inner, _, _ = trajectories[j]
                    if trajectory_type_inner == self.expert_tag:
                        distance = self.calculate_distance_of_trajectories(
                            trajectories[i], trajectories[j]
                        )
                        distances.append(distance)
                        distances_per_irl_trajectory[irl_trajectory_counter].append(
                            distance
                        )
                irl_trajectory_counter += 1
        # print(distances_per_irl_trajectory)
        return distances, distances_per_irl_trajectory

    def histogram_plot(self, ax, distances):
        """
        method to create the histogram of the distances between experts and irl trajectories

        Arguments:
            ax plt.Axes - axes where to plot the histogram
            distances - list of distances to plot the histogram from
        """
        ax.tick_params(labelsize=16)
        ax.hist(distances, bins=20)

    def tsne_plot(self, ax, all_trajectories, agentIndex):
        """Method for creating the tsne plot

        Arguments:
            ax -- pyplot.Axes agent specific axes of the fig
            all_trajectories -- dict dict containing the trajectory lists of all agents
            agentIndex -- index of agent
        """
        print("Create TSNE plot")
        trajectories = all_trajectories["agent" + str(agentIndex)]
        number_of_trajectories = len(trajectories)
        distance_matrix = np.zeros((number_of_trajectories, number_of_trajectories))
        trajectory_colors = []
        for i in range(number_of_trajectories):
            trajectory_type, _, _ = trajectories[i]
            if trajectory_type == self.expert_tag:
                trajectory_colors.append("b")
            else:
                trajectory_colors.append("r")
            for j in range(number_of_trajectories):
                distance = self.calculate_distance_of_trajectories(
                    trajectories[i], trajectories[j]
                )
                distance_matrix[i, j] = distance
        embeddings = TSNE(
            perplexity=15,
            n_components=2,
            metric="precomputed",
            early_exaggeration=2.0,
            learning_rate=30.0,
        ).fit_transform(distance_matrix)
        ax.scatter(embeddings[:, 0], embeddings[:, 1], c=trajectory_colors)


if __name__ == "__main__":
    save_folder = f"/tmp/{getpass.getuser()}/irl/experts"
    pickle_base_path = f"/tmp/{getpass.getuser()}/irl/experts/pickle_files/sc01"
    file_names = os.listdir(pickle_base_path)
    new_file_names = []
    expert_trajectory_messages = []
    for file_name in file_names:
        with open(os.path.join(pickle_base_path, file_name), "rb") as f:
            trajectory_message = pickle.load(f)
        expert_trajectory_messages.append(trajectory_message)

    expert_trajectory_messages = utils.filter_trajectory_list(
        expert_trajectory_messages
    )

    evaluator = Evaluator(
        save_folder=save_folder,
        save_file_name="experts.png",
        episode_length=13,
    )
    evaluator.create(expert_trajectory_messages, None, False, False)
