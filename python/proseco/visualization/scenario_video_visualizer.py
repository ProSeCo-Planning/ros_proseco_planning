import sys
from proseco.visualization.base_visualizer import BaseVisualizer
from matplotlib.figure import Figure
import matplotlib.animation as manimation
import numpy as np
from tqdm import tqdm


class ScenarioVideoVisualizer(BaseVisualizer):
    """
    Visualizes a single scenario.
    """

    def __init__(self, data_path: str):
        """
        Initializes a new instance of the scenario visualizer.

        Parameters
        ----------
        data_path
            path to an evaluator output run directory.
        """
        super(ScenarioVideoVisualizer, self).__init__(
            [0, 100], *self.get_scenario_and_trajectory_paths(data_path)
        )

    def draw(self, fig: Figure):
        pass

    def save(self, path: str, updateCallback=None):
        """
        Draws the figure and saves it under the given path.

        Parameters
        ----------
        path
            the file path where the figure should be saved to (including the figure file name).
        updateCallback
            gets called during the video generation with the current progress (between 0 and 1).
        """

        # create figure
        fig = self._create_fig()

        # Draw (top) trajectory plot
        trajectoryAxis = fig.add_subplot(211)
        trajectoryAxis, agent_plot_objects = self.visualizeTrajectory.drawScenario(
            trajectoryAxis,
            self.scenario_file,
            self.drivenTrajectory,
            annotateAgents=False,
        )
        trajectoryAxis.set_title("Vehicle Movement x-y", fontsize=self.m_textsize)

        # Draw (bottom) velocity plot
        velocityAxis = fig.add_subplot(212)
        velocityAxis = self.visualizeTrajectory.drawVelocityX(
            velocityAxis, self.drivenTrajectory
        )
        # remove the legend
        # velocityAxis.legend_.remove()
        # Draw vertical line into the velocity plot
        currentVelocityIndicator = velocityAxis.plot([0, 0], [-15, 15], "--k")
        currentVelocityIndicator = currentVelocityIndicator[0]

        ###############################################################################
        # Create the video
        ###############################################################################
        numberDataElements = self.drivenTrajectory["numberDataElements"]
        windowWidth = 100
        # Make video a little wider than the vehicles position
        offset = 10

        frontOffset = offset
        xMax = np.max(self.drivenTrajectory["agentsPositionX"][0, :])
        xMin = np.min(self.drivenTrajectory["agentsPositionX"][0, :])
        deltaX = xMax - xMin
        if deltaX < windowWidth:
            extendWindow = (windowWidth - deltaX) / 2
            windowXMin = xMin - extendWindow
            windowXMax = xMax + extendWindow
            trajectoryAxis.set_xlim(windowXMin, windowXMax)
        else:
            trajectoryAxis.set_xlim(xMin - offset, xMax + frontOffset)

        def get_frames():
            for i in range(numberDataElements):  # numberDataElements frames
                # skip time step if duplicate
                if i > 0:
                    if (
                        self.drivenTrajectory["time"][i]
                        == self.drivenTrajectory["time"][i - 1]
                    ):
                        continue
                yield i

        frames = list(get_frames())

        pbar = tqdm(desc="Rendering video", total=len(frames), unit=" frames")

        def animate(i):
            ###############################################################################
            # Update the dynamic objects
            ###############################################################################
            # current vehicle position
            for j, plotObject in enumerate(agent_plot_objects):
                self.visualizeTrajectory.updateObject(
                    trajectoryAxis,
                    plotObject,
                    "vehicle",
                    self.drivenTrajectory["agentsPositionX"][i, j],
                    self.drivenTrajectory["agentsPositionY"][i, j],
                    self.drivenTrajectory["agentsHeading"][i, j],
                    self.drivenTrajectory["agentsVehicleLength"][i, j],
                    self.drivenTrajectory["agentsVehicleWidth"][i, j],
                )

            # current velocity -> vertical bar
            currentVelocityIndicator.set_xdata(
                [self.drivenTrajectory["time"][i], self.drivenTrajectory["time"][i]]
            )

            ###############################################################################
            # Adjust the axis limits
            ###############################################################################
            # Set image view to current extract
            # extract set to be 100m
            xMax = np.max(self.drivenTrajectory["agentsPositionX"][i, :])
            xMin = np.min(self.drivenTrajectory["agentsPositionX"][i, :])
            # Make video a little wider than the vehicles position
            deltaX = xMax - xMin
            if deltaX < windowWidth:
                extendWindow = (windowWidth - deltaX) / 2
                windowXMin = xMin - extendWindow
                windowXMax = xMax + extendWindow
                trajectoryAxis.set_xlim(windowXMin, windowXMax)
            else:
                trajectoryAxis.set_xlim(xMin - offset, xMax + frontOffset)

            pbar.update()
            if updateCallback:
                updateCallback(i / numberDataElements)

        anim = manimation.FuncAnimation(
            fig, animate, frames=frames, cache_frame_data=False
        )
        fps = len(frames) / np.max(self.drivenTrajectory["time"])
        FFwriter = manimation.FFMpegWriter(fps=fps)
        anim.save(path, writer=FFwriter)

        pbar.close()
        fig.clear()


if __name__ == "__main__":
    visualizer = ScenarioVideoVisualizer(sys.argv[1])
    visualizer.save("Scenario.mp4")
