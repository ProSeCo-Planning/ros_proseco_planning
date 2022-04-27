import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union, Tuple

import numpy as np
import rospkg
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from proseco.utility import (
    import_proseco_data,
    visualize_proseco_data,
)
from proseco.utility.io import load_data

# Rospackage path
pack_path = rospkg.RosPack().get_path("ros_proseco_planning")
basis_path = os.path.join(pack_path, "config")


class BaseVisualizer(ABC):
    def __init__(
        self,
        x_axis_limits: Union[list, tuple],
        scenario_path: str,
        trajectory_path: Optional[str],
    ):
        self.scenario_path = scenario_path
        self.scenario_file = load_data(self.scenario_path)
        self.scenario_name = self.scenario_file["name"]

        # Load trajectory file
        if trajectory_path:
            data = load_data(trajectory_path)
            self.drivenTrajectory = import_proseco_data.get_driven_trajectory(data)
        else:
            self.drivenTrajectory = {}

        self.x_axis_limits = x_axis_limits

        # Settings
        self.figureWidth = 20
        self.dpiNumber = 80
        self.faceColor = "w"
        self.edgeColor = "k"
        self.imageFormat = 9 / 16
        # Plot style
        self.m_textsize = 24

        # Initialize trajectory visualizer util object
        numberAgents = len(self.scenario_file["agents"])

        agentsMaxSteeringAngle = np.empty(shape=[numberAgents])
        agentsMaxAbsoluteAcceleration = np.empty(shape=[numberAgents])
        for i in range(0, numberAgents):
            agentsMaxAbsoluteAcceleration[i] = 9.81
            agentsMaxSteeringAngle[i] = import_proseco_data.convert_rad_to_deg(0.22)

        # Save limits to dict
        agentsLimits = {
            "agentsMaxAbsoluteAcceleration": agentsMaxAbsoluteAcceleration,
            "agentsMaxSteeringAngle": agentsMaxSteeringAngle,
        }

        # Create instance of Visualizer class - call the specified methods for creating pre-defined plots
        self.visualizeTrajectory = visualize_proseco_data.TrajectoryVisualizer(
            x_axis_limits,
            agentsLimits,
            self.scenario_file["road"]["number_lanes"],
            self.scenario_file["road"]["lane_width"],
            self.m_textsize,
            "english",
        )
        # self.visualizeTrajectory.m_xMin = np.amin(x_axis_limits)
        # self.visualizeTrajectory.m_xMax = np.amax(x_axis_limits)
        # self.visualizeTrajectory.m_roadXOffset = 0

    @staticmethod
    def get_scenario_and_trajectory_paths(data_path: str) -> Tuple[str, str]:
        """
        Returns the paths to the scenario info and annotated trajectory files.

        Parameters
        ----------
        data_path
            path to the evaluation run directory.

        Returns
        -------
        str
            scenario path.
        str
            trajectory path.
        """
        scenario_path = os.path.join(data_path, "scenario_output.json")
        trajectory_path = str(next(Path(data_path).glob("trajectory_annotated.*")))
        return scenario_path, trajectory_path

    @staticmethod
    def get_default_scenario_path(scenario_name: str) -> str:
        """
        Returns the path to the default scenario info file.

        Parameters
        ----------
        scenario_name
            the name of the scenario.

        Returns
        -------
        str
            scenario path.
        """
        scenario_path = os.path.join(
            basis_path, "scenarios", scenario_name.lower() + ".json"
        )
        return scenario_path

    ##################################################################
    # API METHODS
    ##################################################################

    def save(self, path: str):
        """
        Draws the figure and saves it under the given path.

        Parameters
        ----------
        path
            the file path where the figure should be saved to (including the figure file name).
        """

        # create figure
        fig = self._create_fig()

        # call draw method of overriding class
        self.draw(fig)

        # save figure
        fig.savefig(path, bbox_inches="tight")

    @abstractmethod
    def draw(self, fig: Figure):
        """
        Draws a new subplot onto the figure. This method is intended to be called from either save(self,path)
        or from another visualizer class in its draw(self,fig) method.

        Parameters
        ----------
        fig
            the matplotlib figure.
        """
        pass

    ##################################################################
    # UTILITY METHODS
    ##################################################################

    def _create_fig(self) -> Figure:
        return plt.figure(
            num=self.scenario_name.upper(),
            figsize=(self.figureWidth, self.imageFormat * self.figureWidth),
            dpi=self.dpiNumber,
            facecolor=self.faceColor,
            edgecolor=self.edgeColor,
        )

    def set_limits(self, ax: Axes):
        ax.set_xlim(
            np.min(self.drivenTrajectory["agentsPositionX"][:, 0]) - 20,
            np.max(self.drivenTrajectory["agentsPositionX"][:, 0]) + 20,
        )

    ##################################################################
    # STATIC UTILITY METHODS
    ##################################################################

    @staticmethod
    def list_scenarios() -> List[str]:
        return sorted(
            [
                os.path.splitext(f)[0]
                for f in os.listdir(basis_path + "/scenarios/")
                if f.startswith("sc") and f.endswith(".json")
            ]
        )
