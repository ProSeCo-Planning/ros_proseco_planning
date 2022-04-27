import sys
from matplotlib.figure import Figure
from proseco.visualization.base_visualizer import BaseVisualizer


class RewardVisualizer(BaseVisualizer):
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
        super(RewardVisualizer, self).__init__(
            [0, 1000], *self.get_scenario_and_trajectory_paths(data_path)
        )
        # self.visualizeTrajectory.m_textsize = 12

    def draw(self, fig: Figure):
        maxEgoReward = 500 + 85 + 100
        ax = fig.add_subplot(121)
        ax = self.visualizeTrajectory.drawEgoReward(
            ax, self.drivenTrajectory, maxEgoReward
        )
        ax = fig.add_subplot(122)
        ax = self.visualizeTrajectory.drawCoopReward(ax, self.drivenTrajectory)


if __name__ == "__main__":
    visualizer = RewardVisualizer(sys.argv[1])
    visualizer.save("Reward-plot.svg")
