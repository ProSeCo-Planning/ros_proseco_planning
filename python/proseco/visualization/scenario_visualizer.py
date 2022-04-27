from typing import Optional
from matplotlib.figure import Figure
from proseco.visualization.base_visualizer import BaseVisualizer, basis_path
from tqdm import tqdm
import os


class ScenarioVisualizer(BaseVisualizer):
    """
    Visualizes a single scenario.
    """

    def __init__(
        self,
        scenario_name: str,
        trajectory_path: Optional[str] = None,
        steps: int = 0,
        random_start_pos: bool = True,
    ):
        """
        Initializes a new instance of the scenario visualizer.

        Parameters
        ----------
        scenario_name
            name of the scenario (eg. sc01).
        trajectory_path
            (optional) path to a trajectory_annotated.json for rendering a trajectory.
        steps
            the step that should be visualized.
        """
        super(ScenarioVisualizer, self).__init__(
            [0, 1000], self.get_default_scenario_path(scenario_name), trajectory_path
        )
        self.steps = steps
        self.random_start_pos = random_start_pos

    def draw(self, fig: Figure):
        ax = fig.add_subplot(111)
        # We only want to plot the scenarios between x=[0, 160]. In order to make all vehicles visible,
        # we artifically move those vehicles which are too far away in their initial state.
        # Max_pos_x is the x-position of the agent that is farthest away in the initial state
        max_pos_x = max(
            (agent["vehicle"]["position_x"] for agent in self.scenario_file["agents"])
        )
        for agents in self.scenario_file["agents"]:
            pos_x = agents["vehicle"]["position_x"]
            # if pos_x of agent is further than 150, move the agent close
            if pos_x > 150:
                # move all agents closer by the same amount
                agents["vehicle"]["position_x"] -= max_pos_x - 150

        ax, _ = self.visualizeTrajectory.drawScenario(
            ax,
            self.scenario_file,
            self.drivenTrajectory,
            annotateAgents=False,
            displayRandomStartPos=self.random_start_pos,
        )

        ax.set_xlim([-40, 160])
        # return self.visualizeTrajectory.drawTrajectory2D(fig, ax, self.drivenTrajectory, False)


if __name__ == "__main__":
    scenario_path = os.path.join(basis_path, "scenarios")
    for scenario_name in tqdm(BaseVisualizer.list_scenarios()):
        visualizer = ScenarioVisualizer(scenario_name, random_start_pos=True)
        visualizer.save(os.path.join(scenario_path, f"{scenario_name}.svg"))
