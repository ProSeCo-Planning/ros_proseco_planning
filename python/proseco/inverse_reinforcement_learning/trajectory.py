"""
Translation classes from old rospy messages to the new ray version with json results.
Ideally, this will not be needed in the future when all model and reward classes
are directly accessing the json results from the evaluator.
"""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Action:
    deltaY: float
    deltaVx: float
    likelihoodY: float
    likelihoodVx: float
    selectionLikelihood: float
    muY: float
    muVx: float
    sigmaY: float
    sigmaVx: float
    selectionWeights: List[float]


@dataclass
class State:
    posX: float
    posY: float
    velX: float
    velY: float
    accX: float
    accY: float


@dataclass
class Features:
    diff_vel_vel_des: float
    abs_lane_diff: float
    diff_des_lane_cent: float
    desired_vel: float
    desiredLane: int
    collided: bool
    invalidState: bool
    laneChanged: bool
    invalidAction: bool
    accX: float
    accY: float
    averageAbsoluteAccY: float


@dataclass
class TrajectoryEntry:
    action: Action
    state: State
    features: Features
    featuresOtherAgents: List[Features]

    @staticmethod
    def get_initial_state(agentDic: Dict, lane_width) -> "TrajectoryEntry":
        vehicleDic = agentDic["vehicle"]
        desireDic = agentDic["desire"]

        action = Action(0, 0, 1, 1, 1, 0, 0, 0, 0, [])

        state = State(
            vehicleDic["position_x"],
            vehicleDic["position_y"],
            vehicleDic["velocity_x"],
            vehicleDic["velocity_y"],
            0,
            0,
        )

        desiredVelocity = desireDic["velocity"]
        desiredLane = desireDic["lane"]
        currentLane = vehicleDic["position_y"] // lane_width
        desiredLaneCenter = (currentLane + 0.5) * lane_width
        features = Features(
            desiredVelocity - state.velX,
            abs(0 - desiredLane),
            desiredLaneCenter - state.posY,
            desiredVelocity,
            desiredLane,
            False,
            False,
            False,
            False,
            state.accX,
            state.accY,
            0,
        )

        return TrajectoryEntry(action, state, features, [])


@dataclass
class ScenarioInfo:
    laneWidth: float
    scenarioName: str
    outputPath: str


@dataclass
class Trajectory:
    agentId: int
    initialState: TrajectoryEntry
    trajectory: List[TrajectoryEntry]


@dataclass
class EvaluationRun:
    scenarioInfo: ScenarioInfo
    trajectories: List[Trajectory]
    trajectoryAnnotated: dict

    @staticmethod
    def parse_eval_results(
        irlDic: dict, trajectoryAnnotated: dict, outputPath: str
    ) -> "EvaluationRun":
        """
        Parses the results of the evaluator into the defined data classes.

        Parameters
        ----------
        irlDic
            the dictionary that has been generated by the proseco_planning nodes
        trajectoryAnnotated
            annotated (and more detailed) trajectory dictionary
        outputPath
            disk location where the run has been saved to

        Returns
        -------
        EvaluationRun
            A list of of Trajectory classes where every Trajectory-class represents the trajectory of
            a single agent over the entire run.
        """

        scenarioInfo = ScenarioInfo(
            irlDic["road"]["lane_width"], irlDic["name"], outputPath
        )

        agents: List[Trajectory] = []
        for agentDic in irlDic["agents"]:
            trajectoryEntries: List[TrajectoryEntry] = []
            for trajectoryDic in agentDic["trajectory"]:
                trajectoryEntries.append(
                    TrajectoryEntry(
                        Action(**trajectoryDic["action"]),
                        State(**trajectoryDic["state"]),
                        Features(**trajectoryDic["features"]),
                        [],
                    )
                )

            agents.append(
                Trajectory(
                    agentDic["id"],
                    TrajectoryEntry.get_initial_state(agentDic, scenarioInfo.laneWidth),
                    trajectoryEntries,
                )
            )

        # set features of other agents
        for agent in agents:
            agent.initialState.featuresOtherAgents = [
                a.initialState.features for a in agents if a.agentId != agent.agentId
            ]
            for i, trajectoryEntry in enumerate(agent.trajectory):
                trajectoryEntry.featuresOtherAgents = [
                    a.trajectory[i].features
                    for a in agents
                    if a.agentId != agent.agentId
                ]

        return EvaluationRun(scenarioInfo, agents, trajectoryAnnotated)
