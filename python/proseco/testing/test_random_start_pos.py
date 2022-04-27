import os
from itertools import combinations

import numpy as np
import rospkg
from proseco.testing.tester import TestFailedError
from proseco.utility.visualize_proseco_data import calculateStartPosPolygon
from shapely.geometry import MultiPoint

from proseco.utility.io import load_data

if __name__ == "__main__":
    """
    Tests for all scenarios whether the random start positions of some agents may intersect.
    """

    # Rospackage path
    pack_path = rospkg.RosPack().get_path("ros_proseco_planning")
    basis_path = os.path.join(pack_path, "config")

    scenarios = sorted(
        [
            os.path.splitext(f)[0]
            for f in os.listdir(basis_path + "/scenarios/")
            if f.startswith("sc") and f.endswith(".json")
        ]
    )

    for scenario in scenarios:
        # Read scenario info
        scenario_path = os.path.join(basis_path, "scenarios", scenario + ".json")
        scenarioInfo = load_data(scenario_path)

        polygons = []  # list of possible start positions

        # iterate through all possible start positions
        for points in calculateStartPosPolygon(scenarioInfo):
            # flatten array
            np_points = np.array([x for y in points for x in y])
            # create multi point geometry
            multi_points = MultiPoint(np_points)
            # create polygon from convex hull
            polygons.append(multi_points.convex_hull)

        # pairwise intersection check
        for p1, p2 in combinations(polygons, r=2):
            if p1.intersects(p2):
                raise TestFailedError(
                    f"Scenario {scenario}: random start positions may intersect"
                )
