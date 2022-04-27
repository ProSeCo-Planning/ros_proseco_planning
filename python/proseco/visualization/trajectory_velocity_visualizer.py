import sys
from matplotlib.figure import Figure
from proseco.visualization.base_visualizer import BaseVisualizer


class TrajectoryVelocityVisualizer(BaseVisualizer):
    """
    Visualizes a single scenario.
    """

    def __init__(self, data_path: str, steps: int = 0):
        """
        Initializes a new instance of the scenario visualizer.

        Parameters
        ----------
        data_path
            path to an evaluator output run directory.
        steps
            the step that should be visualized.
        """
        super(TrajectoryVelocityVisualizer, self).__init__(
            [0, 1000], *self.get_scenario_and_trajectory_paths(data_path)
        )
        self.steps = steps

    def draw(self, fig: Figure):
        ax = fig.add_subplot(211)
        ax, _ = self.visualizeTrajectory.drawScenario(
            ax, self.scenario_file, self.drivenTrajectory, annotateAgents=True
        )
        ax = self.visualizeTrajectory.highlightCurrentState(
            ax, self.drivenTrajectory, self.steps
        )
        ax.set_title("Vehicle Movement x-y", fontsize=self.m_textsize)
        self.set_limits(ax)

        ax = fig.add_subplot(223)
        ax = self.visualizeTrajectory.drawVelocityX(ax, self.drivenTrajectory)

        ax = fig.add_subplot(224)
        ax = self.visualizeTrajectory.drawDistanceX(ax, self.drivenTrajectory)


if __name__ == "__main__":
    visualizer = TrajectoryVelocityVisualizer(sys.argv[1])
    visualizer.save("Trajectory-velocity-plot.svg")
