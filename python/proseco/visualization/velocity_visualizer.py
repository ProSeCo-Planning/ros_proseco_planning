import sys
from matplotlib.figure import Figure
from proseco.visualization.base_visualizer import BaseVisualizer


class VelocityVisualizer(BaseVisualizer):
    """
    Visualizes the velocity of a trajectory
    """

    def __init__(self, data_path: str):
        """
        Initializes a new instance of the velocity visualizer.

        Parameters
        ----------
        data_path
            path to an evaluator output run directory.
        """
        super(VelocityVisualizer, self).__init__(
            [0, 1000], *self.get_scenario_and_trajectory_paths(data_path)
        )

    def draw(self, fig: Figure):
        ax = fig.add_subplot(111)
        self.visualizeTrajectory.drawVelocityX(ax, self.drivenTrajectory)


if __name__ == "__main__":
    visualizer = VelocityVisualizer(sys.argv[1])
    visualizer.save("Velocity-plot.svg")
