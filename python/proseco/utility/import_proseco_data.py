"""
Port of import_proseco_data.py that uses the JSON format
"""

from pathlib import Path
from typing import Any, Dict, Union
import numpy as np

from proseco.utility.io import load_data


def convert_rad_to_deg(radiants: float) -> float:
    """Converts radiants to degrees.

    Parameters
    ----------
    radiants : float
        The radiants to convert.

    Returns
    -------
    float
        The converted radiants, in degrees.
    """
    return radiants * 180 / np.pi


def get_static_obstacles(data: Dict[str, Any]) -> np.ndarray:
    """Converts the data from the JSON format to the format used by the visualization.

    Parameters
    ----------
    data : [type]
        The data loaded from the JSON file.

    Returns
    -------
    np.ndarray
        The obstacle data in the format used by the visualization, containing all information required for plotting the obstacles in the scenario.
    """
    # Read data from file
    # data = load_data(data)

    if "obstacles" not in data or data["obstacles"] is None:
        return []

    # Implement data import for m obstacles with for-loop
    numberObstacles = len(data["obstacles"])

    # initialize matrizes for collecting the data
    obstacleDistanceX = np.empty(shape=[numberObstacles])
    obstacleDistanceY = np.empty(shape=[numberObstacles])
    obstacleHeading = np.empty(shape=[numberObstacles])
    obstacleLength = np.empty(shape=[numberObstacles])
    obstacleWidth = np.empty(shape=[numberObstacles])

    # collect the data for obstacles
    for i in range(0, numberObstacles):
        # Extract data from import
        obstacle = data["obstacles"][i]
        obstacleDistanceX[i] = obstacle["position_x"]
        obstacleDistanceY[i] = obstacle["position_y"]
        obstacleLength[i] = obstacle["length"]
        obstacleWidth[i] = obstacle["width"]
        obstacleHeading[i] = obstacle["heading"]

    obstacleData = np.empty(shape=[numberObstacles, 5])
    obstacleData[:, 0] = obstacleDistanceX
    obstacleData[:, 1] = obstacleDistanceY
    obstacleData[:, 2] = obstacleHeading
    obstacleData[:, 3] = obstacleLength
    obstacleData[:, 4] = obstacleWidth

    return obstacleData


def get_driven_trajectory(data: Dict[str, Any]) -> Dict[str, Any]:
    """Converts the data from the JSON format to the format used by the visualization.

    Parameters
    ----------
    data : Dict[str, Any]
        The data loaded from the JSON file.

    Returns
    -------
    Dict[str, Any]
        The trajectory data in the format used by the visualization, containing all information required for plotting the trajectory such as x,y position velocities etc. for all agents
    """
    # data = load_data(input)
    # collect Data for return
    trajectoryData = {}
    if len(data) == 0:
        print("Empty single shot plan")
        trajectoryData["corruptedDataSet"] = True

        return trajectoryData

    # Implement data import for n agents and m obstacles with for-loop
    numberAgents = len(data["agents"])
    numberObstacles = (
        0
        if "obstacles" not in data or data["obstacles"] is None
        else len(data["obstacles"])
    )
    numberDataElements = len(data["agents"][0]["trajectory"])

    stage = [x["step"] for x in data["agents"][0]["trajectory"]]
    numberStages = np.max(stage) - np.min(stage) + 1
    time = [x["time"] for x in data["agents"][0]["trajectory"]]

    # initialize matrizes for collecting the data
    agentsID = np.empty(shape=[numberDataElements, numberAgents])
    agentsPositionX = np.empty(shape=[numberDataElements, numberAgents])
    agentsPositionY = np.empty(shape=[numberDataElements, numberAgents])
    agentsLane = np.empty(shape=[numberDataElements, numberAgents])
    agentsPositionYDesired = np.empty(shape=[numberDataElements, numberAgents])
    agentsVelocityX = np.empty(shape=[numberDataElements, numberAgents])
    agentsVelocityDesiredX = np.empty(shape=[numberDataElements, numberAgents])
    agentsVelocityY = np.empty(shape=[numberDataElements, numberAgents])
    agentsAccelerationX = np.empty(shape=[numberDataElements, numberAgents])
    agentsAccelerationY = np.empty(shape=[numberDataElements, numberAgents])
    agentsHeading = np.empty(shape=[numberDataElements, numberAgents])
    agentsTotalAcceleration = np.empty(shape=[numberDataElements, numberAgents])
    agentsSteeringAngle = np.empty(shape=[numberDataElements, numberAgents])
    agentsEgoReward = np.empty(shape=[numberDataElements, numberAgents])
    agentsCoopReward = np.empty(shape=[numberDataElements, numberAgents])
    agentsVehicleWidth = np.empty(shape=[numberDataElements, numberAgents])
    agentsVehicleLength = np.empty(shape=[numberDataElements, numberAgents])
    agentsPredefined = np.empty(shape=[numberAgents])

    obstaclesID = np.empty(shape=[numberDataElements, numberObstacles])
    obstaclesDistanceX = np.empty(shape=[numberDataElements, numberObstacles])
    obstaclesDistanceY = np.empty(shape=[numberDataElements, numberObstacles])
    obstaclesHeading = np.empty(shape=[numberDataElements, numberObstacles])
    obstaclesVehicleWidth = np.empty(shape=[numberDataElements, numberObstacles])
    obstaclesVehicleLength = np.empty(shape=[numberDataElements, numberObstacles])

    # collect the data for agents
    for i in range(0, numberAgents):
        agent = data["agents"][i]
        vehicle = agent["vehicle"]
        desire = agent["desire"]
        trajectory = agent["trajectory"]

        def select_trajectory_elem(key):
            return [x[key] for x in trajectory]

        # Extract data from import
        m_id = [agent["id"]] * numberDataElements
        m_CoordinateYDesired = [desire["lane"]] * numberDataElements
        m_velocityDesiredX = [desire["velocity"]] * numberDataElements
        m_width = [vehicle["width"]] * numberDataElements
        m_length = [vehicle["length"]] * numberDataElements
        agentsPredefined[i] = agent["is_predefined"]
        m_lane = select_trajectory_elem("lane")
        m_positionX = select_trajectory_elem("position_x")
        m_positionY = select_trajectory_elem("position_y")
        m_velocityX = select_trajectory_elem("velocity_x")
        m_velocityY = select_trajectory_elem("velocity_y")
        m_accelerationX = select_trajectory_elem("acceleration_x")
        m_accelerationY = select_trajectory_elem("acceleration_y")
        m_heading = select_trajectory_elem("heading")
        m_totalAcceleration = select_trajectory_elem("total_acceleration")
        m_steeringAngle = [0] * numberDataElements  # TODO
        m_egoReward = select_trajectory_elem("ego_reward")
        m_coopReward = select_trajectory_elem("coop_reward")

        # Convert rad to degree
        for j in range(0, numberDataElements):
            m_steeringAngle[j] = convert_rad_to_deg(m_steeringAngle[j])

        agentsID[:, i] = m_id
        agentsPositionX[:, i] = m_positionX
        agentsPositionY[:, i] = m_positionY
        agentsLane[:, i] = m_lane
        agentsPositionYDesired[:, i] = m_CoordinateYDesired
        agentsVelocityX[:, i] = m_velocityX
        agentsVelocityDesiredX[:, i] = m_velocityDesiredX
        agentsVelocityY[:, i] = m_velocityY
        agentsAccelerationX[:, i] = m_accelerationX
        agentsAccelerationY[:, i] = m_accelerationY
        agentsHeading[:, i] = m_heading
        agentsTotalAcceleration[:, i] = m_totalAcceleration
        agentsSteeringAngle[:, i] = m_steeringAngle
        agentsEgoReward[:, i] = m_egoReward
        agentsCoopReward[:, i] = m_coopReward
        agentsVehicleWidth[:, i] = m_width
        agentsVehicleLength[:, i] = m_length

    # collect the data for obstacles
    for i in range(0, numberObstacles):
        # Extract data from import
        obstacle = data["obstacles"][i]
        obstaclesID[:, i] = obstacle["id"]
        obstaclesDistanceX[:, i] = obstacle["position_x"]
        obstaclesDistanceY[:, i] = obstacle["position_y"]
        obstaclesVehicleLength[:, i] = obstacle["length"]
        obstaclesVehicleWidth[:, i] = obstacle["width"]
        obstaclesHeading[:, i] = obstacle["heading"]

    numberLanes = data["road"]["number_lanes"]
    laneWidth = data["road"]["lane_width"]

    # Indicate if single shot plan was not explored
    trajectoryData["corruptedDataSet"] = False

    # Common data
    trajectoryData["stage"] = stage
    trajectoryData["numberAgents"] = numberAgents
    trajectoryData["numberObstacles"] = numberObstacles
    trajectoryData["numberDataElements"] = numberDataElements
    trajectoryData["numberStages"] = numberStages
    trajectoryData["time"] = time

    # Agent specific data
    trajectoryData["agentsID"] = agentsID
    trajectoryData["agentsPositionX"] = agentsPositionX
    trajectoryData["agentsPositionY"] = agentsPositionY
    trajectoryData["agentsLane"] = agentsLane
    trajectoryData["agentsPositionYDesired"] = agentsPositionYDesired
    trajectoryData["agentsVelocityX"] = agentsVelocityX
    trajectoryData["agentsVelocityDesiredX"] = agentsVelocityDesiredX
    trajectoryData["agentsVelocityY"] = agentsVelocityY
    trajectoryData["agentsAccelerationX"] = agentsAccelerationX
    trajectoryData["agentsAccelerationY"] = agentsAccelerationY
    trajectoryData["agentsHeading"] = agentsHeading
    trajectoryData["agentsTotalAcceleration"] = agentsTotalAcceleration
    trajectoryData["agentsSteeringAngle"] = agentsSteeringAngle
    trajectoryData["agentsEgoReward"] = agentsEgoReward
    trajectoryData["agentsCoopReward"] = agentsCoopReward
    trajectoryData["agentsVehicleWidth"] = agentsVehicleWidth
    trajectoryData["agentsVehicleLength"] = agentsVehicleLength
    trajectoryData["agentsPredefined"] = agentsPredefined

    # Obstacle specific data
    trajectoryData["obstaclesID"] = obstaclesID
    trajectoryData["obstaclesDistanceX"] = obstaclesDistanceX
    trajectoryData["obstaclesDistanceY"] = obstaclesDistanceY
    trajectoryData["obstaclesHeading"] = obstaclesHeading
    trajectoryData["obstaclesVehicleWidth"] = obstaclesVehicleWidth
    trajectoryData["obstaclesVehicleLength"] = obstaclesVehicleLength

    # Road specific data
    trajectoryData["numberLanes"] = numberLanes
    trajectoryData["laneWidth"] = laneWidth

    return trajectoryData
