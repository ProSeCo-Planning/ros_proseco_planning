#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# \author  Florian Engelhorn <engelhor@fzi.de>
# \date    2017-12-12
#


###############################################################################
# This script provides functions which are used to visualize the ProSeCo data
###############################################################################


from typing import Any, Dict, List, Optional, Tuple
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Polygon
from matplotlib.axes import Axes
from scipy.spatial import ConvexHull

# import functions for ProSeCo trajectory specific import
from proseco.utility import import_proseco_data

# Apply tex style

# Tex style
plt.rc("font", family="serif")

baseFontSize = 40  # 18

params = {
    "axes.labelsize": baseFontSize,
    "font.size": baseFontSize,
    "legend.fontsize": baseFontSize,
    "xtick.labelsize": baseFontSize - 2,
    "ytick.labelsize": baseFontSize - 2,
    "text.usetex": True,
    "pdf.fonttype": 42,
    "figure.figsize": [9, 5],
}
plt.rcParams.update(params)


def calculateDensity(xGrid, yGrid, xSample, ySample) -> np.ndarray:
    """

    Parameters
    ----------
    xGrid
        x-values of evaluation grid
    yGrid
        y-values of evaluation grid
    xSample
        x-values of sample points
    ySample
        y-values of sample points

    Returns
    -------
    values : np.ndarray
        density value at each grid value: (dimension xGrid.size * yGrid.size)

    """
    density = np.empty(shape=[xGrid.size])
    xTolerance = 1.0
    yTolerance = 1.0

    for i in range(0, xGrid.size):
        tempDensity = 0
        for k in range(0, xSample.size):
            if (
                xSample[k] <= xGrid[i] + xTolerance
                and xSample[k] >= xGrid[i] - xTolerance
                and ySample[k] <= yGrid[i] + yTolerance
                and ySample[k] >= yGrid[i] - yTolerance
            ):
                tempDensity += 1

        density[i] = tempDensity / xTolerance / yTolerance

    return density


def calcRectangleCoordinates(
    vehicleX: float,
    vehicleY: float,
    vehicleHeading: float,
    vehicleLength: float,
    vehicleWidth: float,
) -> Tuple[list, list]:
    """
    Calculates the rectangle bounding box position of an object (eg. vehicle or obstacle)

    Parameters
    ----------
    vehicleX
        x position of the object
    vehicleY
        y position of the object
    vehicleHeading
        heading of the object
    vehicleLength
        length of the object
    vehicleWidth
        width of the object

    Returns
    -------
    x
        x-coordinates
    y
        y-coordinates
    """
    # Specify vehicle
    # p1: Front left
    # p2: Front right
    # p3: Back left
    # p4: Back right

    p1x = (
        vehicleX
        + vehicleLength * np.cos(vehicleHeading)
        - vehicleWidth / 2 * np.sin(vehicleHeading)
    )
    p1y = (
        vehicleY
        + vehicleLength * np.sin(vehicleHeading)
        + vehicleWidth / 2 * np.cos(vehicleHeading)
    )

    p2x = (
        vehicleX
        + vehicleLength * np.cos(vehicleHeading)
        + vehicleWidth / 2 * np.sin(vehicleHeading)
    )
    p2y = (
        vehicleY
        + vehicleLength * np.sin(vehicleHeading)
        - vehicleWidth / 2 * np.cos(vehicleHeading)
    )

    p3x = vehicleX - vehicleWidth / 2 * np.sin(vehicleHeading)
    p3y = vehicleY + vehicleWidth / 2 * np.cos(vehicleHeading)

    p4x = vehicleX + vehicleWidth / 2 * np.sin(vehicleHeading)
    p4y = vehicleY - vehicleWidth / 2 * np.cos(vehicleHeading)

    xData = [p1x, p2x, p4x, p3x, p1x]
    yData = [p1y, p2y, p4y, p3y, p1y]

    return xData, yData


def calcTriangleCoordinates(
    vehicleX: float,
    vehicleY: float,
    vehicleHeading: float,
    vehicleLength: float,
    vehicleWidth: float,
) -> Tuple[list, list]:
    """
    Draw rectangle for indicating the orientation of the vehicle

    Parameters
    ----------
    vehicleX
        x position of the object
    vehicleY
        y position of the object
    vehicleHeading
        heading of the object
    vehicleLength
        length of the object
    vehicleWidth
        width of the object

    Returns
    -------
    x
        x-coordinates
    y
        y-coordinates
    """
    # p1: mid of front axle
    p1xr = vehicleX + vehicleLength * np.cos(vehicleHeading)
    p1yr = vehicleY + vehicleLength * np.sin(vehicleHeading)
    # p2: left
    p2xr = (
        vehicleX
        - vehicleWidth / 2 * np.sin(vehicleHeading)
        + vehicleLength * 3 / 4 * np.cos(vehicleHeading)
    )
    p2yr = (
        vehicleY
        + vehicleWidth / 2 * np.cos(vehicleHeading)
        + vehicleLength * 3 / 4 * np.sin(vehicleHeading)
    )
    # p3: right
    p3xr = (
        vehicleX
        + vehicleWidth / 2 * np.sin(vehicleHeading)
        + vehicleLength * 3 / 4 * np.cos(vehicleHeading)
    )
    p3yr = (
        vehicleY
        - vehicleWidth / 2 * np.cos(vehicleHeading)
        + vehicleLength * 3 / 4 * np.sin(vehicleHeading)
    )

    xPointsTriangle = [p1xr, p2xr, p3xr, p1xr]
    yPointsTriangle = [p1yr, p2yr, p3yr, p1yr]

    return xPointsTriangle, yPointsTriangle


def calculateStartPosPolygon(scenarioInfo: Dict[str, Any]) -> List[list]:
    """
    Calculates all possible start positions and returns the convex hull of it.

    Parameters
    ----------
    scenarioInfo
        the scenario json object

    Returns
    -------
    coordinates
        list of tuples with x and y coordinates
    """
    result = []
    for id_a, agent in enumerate(scenarioInfo["agents"]):
        if not agent["vehicle"]["random"]:
            continue
        pos_x = agent["vehicle"]["position_x"]
        pos_y = agent["vehicle"]["position_y"]
        length = agent["vehicle"]["length"]
        width = agent["vehicle"]["width"]
        heading = agent["vehicle"]["heading"]
        sigma_x = agent["vehicle"]["sigma_position_x"]
        sigma_y = agent["vehicle"]["sigma_position_y"]
        sigma_heading = agent["vehicle"]["sigma_heading"]
        if sigma_x == 0 or sigma_y == 0:
            continue

        def calcCoordinates(x, y, heading):
            return [
                list(x)
                for x in zip(*calcRectangleCoordinates(x, y, heading, length, width))
            ]

        # top left
        p1 = calcCoordinates(
            pos_x - sigma_x * 2, pos_y + sigma_y * 2, heading + sigma_heading * 2
        )
        # top right
        p2 = calcCoordinates(
            pos_x + sigma_x * 2, pos_y + sigma_y * 2, heading + sigma_heading * 2
        )
        # bottom left
        p3 = calcCoordinates(
            pos_x - sigma_x * 2, pos_y - sigma_y * 2, heading - sigma_heading * 2
        )
        # bottom right
        p4 = calcCoordinates(
            pos_x + sigma_x * 2, pos_y - sigma_y * 2, heading - sigma_heading * 2
        )

        result.append([p1, p2, p3, p4])
    return result


class TrajectoryVisualizer:
    """A class used as visualizer of trajectory related data"""

    ## Evaluation Details
    m_numberAgents = None
    ## Axis Limits
    m_xMin = None
    m_xMax = None
    m_yMin = None
    m_yMax = None

    # value used for displaying greater area as specified by min/max values
    m_offset = 0.25

    # Extension of road
    m_roadXOffset = 50

    # road object
    m_road = None

    ## Font size for plots
    m_textsize = 16
    m_symbolsize = 8
    m_scatterSymbolSize = 50
    m_linestyles = [
        "x",
        "o",
        "s",
        "o",
        "v",
        "^",
        "<",
        ">",
        "8",
        "s",
        "p",
        "*",
        "h",
        "H",
        "D",
        "d",
        "P",
        "X",
    ]
    # m_colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
    m_colors = [
        "#007749",
        "#c10033",
        "#0065a3",
        "#ffb800",
        "#ec9d27",
        "#87c3e7",
        "#000000",
        "#f3c3b5",
    ]
    base_color_expert = "#36688D"
    base_color_irl = "#F18904"
    m_colors_experts = [
        "#ff8080",
        "#ff8080",
        "#ff8080",
        "#ffb800",
        "#ec9d27",
        "#87c3e7",
        "#000000",
        "#f3c3b5",
    ]
    m_colors_irl = [
        "#0000ff",
        "#0000ff",
        "#0000ff",
        "#0000ff",
        "#0065a3",
        "#ffb800",
        "#000000",
        "#f3c3b5",
    ]
    m_colorMap = cm.winter
    m_timeColor = cm.cool
    m_language = "english"

    def __init__(
        self,
        agentsPositionX,
        agentsLimits,
        numberLanes,
        laneWidth,
        m_textsize,
        language="german",
    ):
        ## store physical limitation of actions
        self.m_agentsLimits = agentsLimits
        ## store road properties
        self.m_numberLanes = numberLanes
        self.m_laneWidth = laneWidth
        ## Determine xLimits
        self.m_xMin = np.amin(agentsPositionX) - self.m_roadXOffset
        self.m_xMax = np.amax(agentsPositionX) + self.m_roadXOffset
        ## Determine yLimits
        self.m_yMin = -self.m_offset
        self.m_yMax = self.m_numberLanes * self.m_laneWidth + self.m_offset

        ## Plot options
        self.m_textsize = m_textsize

        self.m_language = language

    def drawObject(
        self,
        ax: Axes,
        objectType: str,
        vehicleX: float,
        vehicleY: float,
        vehicleHeading: float,
        vehicleLength: float,
        vehicleWidth: float,
        color="#000000",
    ) -> Tuple[Axes, dict]:
        """
        Draws a rectangle (the vehicle/obstacle) accordingly to its position and orientation

        Parameters
        ----------
        ax
            the matplotlib axes
        objectType
            either "vehicle" or "obstacle"
        vehicleX
            x position of the object
        vehicleY
            y position of the object
        vehicleHeading
            heading of the object
        vehicleLength
            length of the object
        vehicleWidth
            width of the object
        color
            color of the object

        Returns
        -------
        ax
            matplotlib Axes
        dictionary
            `{"rectangleObject": <rectangle bounding box>, "triangleObject": <triangle, if objectType=="vehicle">}`
        """

        # Calculate the corner points of the vehicle
        xPoints, yPoints = calcRectangleCoordinates(
            vehicleX, vehicleY, vehicleHeading, vehicleLength, vehicleWidth
        )
        rectangleObject = ax.add_patch(
            Polygon(
                [[x, y] for x, y in zip(xPoints, yPoints)],
                closed=True,
                fill=False,
                color=color,
            )
        )

        # initialize object
        triangleObject = None
        if objectType == "vehicle":
            # Calculate the corner points of the triangle
            xTrianglePoints, yTrianglePoints = calcTriangleCoordinates(
                vehicleX, vehicleY, vehicleHeading, vehicleLength, vehicleWidth
            )
            triangleObject = ax.add_patch(
                Polygon(
                    [[x, y] for x, y in zip(xTrianglePoints, yTrianglePoints)],
                    closed=True,
                    fill=False,
                    color=color,
                )
            )
        elif objectType == "obstacle":
            ## Create some nice hashing for indicating a static obstacle
            rectangleObject.set_hatch("///")

        else:
            print("Wrong object type specified within drawObject")

        # Collect data for export
        vehicleData = {}
        vehicleData["rectangleObject"] = rectangleObject
        vehicleData["triangleObject"] = triangleObject

        return ax, vehicleData

    def updateObject(
        self,
        ax: Axes,
        objectData: dict,
        objectType: str,
        vehicleX,
        vehicleY,
        vehicleHeading,
        vehicleLength,
        vehicleWidth,
    ) -> Axes:
        """
        Updates a matplotlib object with new position and heading values.

        Parameters
        ----------
        ax
            the matplotlib axes
        objectData
            dictionary containng the matplotlib objects to update (returned from drawObject())
        objectType
            either "vehicle" or "obstacle"
        vehicleX
            x position of the object
        vehicleY
            y position of the object
        vehicleHeading
            heading of the object
        vehicleLength
            length of the object
        vehicleWidth
            width of the object

        Returns
        -------
        ax
            the matplotlib axes
        """
        currentRectangleData = objectData["rectangleObject"]
        xData, yData = calcRectangleCoordinates(
            vehicleX, vehicleY, vehicleHeading, vehicleLength, vehicleWidth
        )
        currentRectangleData.set_xy([[x, y] for x, y in zip(xData, yData)])

        currentTriangleData = objectData["triangleObject"]
        if currentTriangleData != None:
            xTrianglePoints, yTrianglePoints = calcTriangleCoordinates(
                vehicleX, vehicleY, vehicleHeading, vehicleLength, vehicleWidth
            )
            currentTriangleData.set_xy(
                [[x, y] for x, y in zip(xTrianglePoints, yTrianglePoints)]
            )

        return ax

    def drawStaticObstacles(self, ax: Axes, obstacleData):
        """
        Draws static obstacles from an array of obstacles.

        *Deprecated* use drawObstacles() instead.

        Parameters
        ----------
        ax
            the matplotlib axes
        obstacleData
            2d array with obstacle positions

        Returns
        -------
        ax
            the matplotlib axes
        """
        numberObstacles = np.shape(obstacleData)[0]
        for i in range(0, numberObstacles):
            self.drawObject(
                ax,
                "obstacle",
                obstacleData[i, 0],
                obstacleData[i, 1],
                obstacleData[i, 2],
                obstacleData[i, 3],
                obstacleData[i, 4],
            )

        return ax

    def drawObstacles(self, ax: Axes, scenarioInfo: dict) -> Axes:
        """
        Draws static obstacles from a scenario json file.

        Parameters
        ----------
        ax
            the matplotlib axes
        scenarioInfo
            the scenario json object

        Returns
        -------
        ax
            the matplotlib axes
        """
        for obstacle in scenarioInfo["obstacles"] or []:
            ax, _ = self.drawObject(
                ax,
                "obstacle",
                obstacle["position_x"],
                obstacle["position_y"],
                obstacle["heading"],
                obstacle["length"],
                obstacle["width"],
            )
        return ax

    def drawRoad(self, ax: Axes) -> Axes:
        """
        Draw the road within a x-y-trajectory plot

        Parameters
        ----------
        ax
            the matplotlib axes

        Returns
        -------
        ax
            the matplotlib axes
        """

        # Specification of road geometry
        # extend road

        xLimitsRoad = [
            self.m_xMin - self.m_roadXOffset,
            self.m_xMax + self.m_roadXOffset,
        ]

        roadWidth = self.m_numberLanes * self.m_laneWidth

        # plot details
        laneCenterColor = (168 / 255, 168 / 255, 168 / 255)
        lineWidth = 2

        # Outer most lane markers
        ax.plot(
            xLimitsRoad, [0, 0], linestyle="-", color="k", linewidth=lineWidth
        )  # solid
        ax.plot(
            xLimitsRoad,
            [roadWidth, roadWidth],
            linestyle="-",
            color="k",
            linewidth=lineWidth,
        )
        for i in range(0, self.m_numberLanes + 1):
            if i == 0 | i == self.m_numberLanes + 1:
                # Road boundaries
                style = "-"
            else:
                # Inner lane markers
                style = "--"

            yCoor = i * self.m_laneWidth

            ax.plot(
                xLimitsRoad,
                [yCoor, yCoor],
                linestyle=style,
                color="k",
                linewidth=lineWidth,
            )

            # Lane center
            if i != 0:
                laneCenter = yCoor - self.m_laneWidth / 2
                ax.plot(
                    xLimitsRoad,
                    [laneCenter, laneCenter],
                    linestyle=":",
                    color=laneCenterColor,
                    linewidth=lineWidth,
                )

        return ax

    def addStartPointOfActionSingleAgent(
        self,
        ax: Axes,
        agentIndex: int,
        currentTrajectory: Dict[str, Any],
        formatInputData: str,
        yData: str = "agentsPositionY",
    ) -> Axes:
        """
        Calculate the starting point where a new action has been chosen and draws it to given axes

        Parameters
        ----------
        ax
            the matplotlib axes
        currentTrajectory
            trajectory data
        formatInputData
            either "time", "2D", "3D" or "singleShot".
        yData
            key in currentTrajectory for selecting the y-data. Only used for formatInputData=="2D"

        Returns
        -------
        ax
            the matplotlib axes
        """

        nPointsPerAction = int(
            currentTrajectory["numberDataElements"]
            / (currentTrajectory["numberStages"])
        )
        j = agentIndex
        i = 0

        if formatInputData == "time":
            ax.plot(
                currentTrajectory["time"][i * nPointsPerAction],
                currentTrajectory[yData][i * nPointsPerAction, j],
                marker=self.m_linestyles[j],
                color="k",
            )
        elif formatInputData == "2D":
            ax.plot(
                currentTrajectory["agentsPositionX"][i * nPointsPerAction, j],
                currentTrajectory["agentsPositionY"][i * nPointsPerAction, j],
                marker=self.m_linestyles[j],
                color="k",
            )
        elif formatInputData == "3D":
            ax.scatter(
                currentTrajectory["agentsPositionX"][i * nPointsPerAction, j],
                currentTrajectory["agentsPositionY"][i * nPointsPerAction, j],
                currentTrajectory["time"][i * nPointsPerAction],
                marker=self.m_linestyles[j],
                color="k",
            )
        elif formatInputData == "singleShot":
            ax.scatter(
                currentTrajectory["agentsPositionX"][i * nPointsPerAction, j],
                currentTrajectory["agentsPositionY"][i * nPointsPerAction, j],
                marker=self.m_linestyles[j],
                color="k",
                s=self.m_scatterSymbolSize - 10,
            )

        return ax

    def addStartPointOfAction(
        self,
        ax: Axes,
        currentTrajectory: Dict[str, Any],
        formatInputData: str,
        yData: str = "agentsPositionY",
    ) -> Axes:
        """
        Calculate the points where a new action has been chosen and draw it to given axes

        Parameters
        ----------
        ax
            the matplotlib axes
        currentTrajectory
            trajectory data
        formatInputData
            either "time", "2D", "3D" or "singleShot".
        yData
            key in currentTrajectory for selecting the y-data. Only used for formatInputData=="2D"

        Returns
        -------
        ax
            the matplotlib axes
        """
        nPointsPerAction = int(
            currentTrajectory["numberDataElements"]
            / (currentTrajectory["numberStages"])
        )

        for i in range(0, currentTrajectory["numberStages"]):
            for j in range(0, currentTrajectory["numberAgents"]):
                if formatInputData == "time":
                    ax.plot(
                        currentTrajectory["time"][i * nPointsPerAction],
                        currentTrajectory[yData][i * nPointsPerAction, j],
                        marker=self.m_linestyles[j],
                        color="k",
                    )
                elif formatInputData == "2D":
                    ax.plot(
                        currentTrajectory["agentsPositionX"][i * nPointsPerAction, j],
                        currentTrajectory["agentsPositionY"][i * nPointsPerAction, j],
                        marker=self.m_linestyles[j],
                        color="k",
                    )
                elif formatInputData == "3D":
                    ax.scatter(
                        currentTrajectory["agentsPositionX"][i * nPointsPerAction, j],
                        currentTrajectory["agentsPositionY"][i * nPointsPerAction, j],
                        currentTrajectory["time"][i * nPointsPerAction],
                        marker=self.m_linestyles[j],
                        color="k",
                    )
                elif formatInputData == "singleShot":
                    ax.scatter(
                        currentTrajectory["agentsPositionX"][i * nPointsPerAction, j],
                        currentTrajectory["agentsPositionY"][i * nPointsPerAction, j],
                        marker=self.m_linestyles[j],
                        color="k",
                        s=self.m_scatterSymbolSize - 10,
                    )

        return ax

    def drawTrajectory3D(self, ax: Axes, currentTrajectory: Dict[str, Any]) -> Axes:
        """
        Draws the x-y-t movement in 3D for all agents

        Parameters
        ----------
        ax
            the matplotlib axes
        currentTrajectory
            trajectory data

        Returns
        -------
        ax
            the matplotlib axes
        """

        # x-y-t movement 3D
        for i in range(0, currentTrajectory["numberAgents"]):
            name = "agent: " + str(int(currentTrajectory["agentsID"][0, i]))
            ax.plot3D(
                currentTrajectory["agentsPositionX"][:, i],
                currentTrajectory["agentsPositionY"][:, i],
                currentTrajectory["time"],
                "o",
                label=name,
                color=self.m_colors[i % len(self.m_colors)],
            )
            ax.plot3D(
                currentTrajectory["agentsPositionX"][:, i],
                currentTrajectory["agentsPositionYDesired"][:, i],
                currentTrajectory["time"],
                label=name + " desired",
                color=self.m_colors[i % len(self.m_colors)],
            )
        # Add start points of actions
        ax = self.addStartPointOfAction(ax, currentTrajectory, "3D")
        # Comment out if plot of road not desired
        self.drawRoad(ax)
        ax.legend(fontsize=self.m_textsize)
        ax.legend(fontsize=self.m_textsize)
        ax.set_xlabel("Position x [m]", fontsize=self.m_textsize)
        ax.set_ylabel("Position y [m]", fontsize=self.m_textsize)
        ax.set_xlim(self.m_xMin, self.m_xMax)
        ax.set_ylim(self.m_yMin, self.m_yMax)
        if self.m_language == "german":
            ax.set_title("$x$-$y$-Fahrzeugbewegung", fontsize=self.m_textsize)
        else:
            ax.set_title("x y vehicle movement", fontsize=self.m_textsize)
        ax.grid(True)
        # plt.show()
        return ax

    def drawTrajectory2DSingleAgent(
        self,
        fig: Figure,
        ax: Axes,
        agentIndex: int,
        currentTrajectory: Dict[str, Any],
        forExperts: bool,
        plotLegend: bool = True,
    ) -> Axes:
        """
        Draws the x-y movement in 2D for a single agent

        TODO: remove

        Parameters
        ----------
        fig
            the matplotlib figure
        ax
            the matplotlib axes
        agentIndex
            the index of the agent to draw
        currentTrajectory
            trajectory data
        forExperts
            true, if the colors should be expert colors
        plotLegend
            true, if the legend should be drawn

        Returns
        -------
        ax
            the matplotlib axes
        """

        if forExperts:
            colors = self.m_colors_experts
        else:
            colors = self.m_colors_irl
        # x-y-Movement 2D
        i = agentIndex
        cf = ax.plot(
            currentTrajectory["agentsPositionX"][:, i],
            currentTrajectory["agentsPositionY"][:, i],
            color=colors[i % len(colors)],
        )
        # Add start points of actions
        ax = self.addStartPointOfActionSingleAgent(
            ax, agentIndex, currentTrajectory, "2D"
        )
        # cb = fig.colorbar(cf, ax=ax)
        if self.m_language == "german":
            # cb.set_label('Zeit $[s]$',fontsize=self.m_textsize)
            ax.set_title("$x$-$y$-$t$ Fahrzeugbewegung", fontsize=self.m_textsize)
        else:
            # cb.set_label('time $[s]$',fontsize=self.m_textsize)
            ax.set_title("Vehicle Movement x-y-t", fontsize=self.m_textsize)
        # Comment out if plot of road not desired
        self.drawRoad(ax)
        if plotLegend:
            ax.legend(
                bbox_to_anchor=(0, 1.02, 1, 0.2),
                loc="lower left",
                mode="expand",
                borderaxespad=0,
                ncol=currentTrajectory["numberAgents"],
            )
        ax.set_xlabel("Position x [m]", fontsize=self.m_textsize)
        ax.set_ylabel("Position y [m]", fontsize=self.m_textsize)
        ax.tick_params(axis="both", which="major", labelsize=self.m_textsize)
        ax.tick_params(axis="both", which="minor", labelsize=self.m_textsize)

        ax.set_xlim(self.m_xMin, self.m_xMax)
        ax.set_ylim(self.m_yMin, self.m_yMax)
        ax.grid(True)

        return ax

    def drawTrajectory2D(
        self,
        fig: Figure,
        ax: Axes,
        currentTrajectory: Dict[str, Any],
        plotLegend: bool = True,
    ) -> Axes:
        """
        Draws the x-y movement in 2D for all agents

        TODO: merge with drawScenario, drawTrajectory2DSingleAgent

        Parameters
        ----------
        fig
            the matplotlib figure
        ax
            the matplotlib axes
        currentTrajectory
            trajectory data
        plotLegend
            true, if the legend should be drawn

        Returns
        -------
        ax
            the matplotlib axes
        """
        for i in range(0, currentTrajectory["numberAgents"]):
            name = "agent: " + str(int(currentTrajectory["agentsID"][0, i]))
            nameDesired = name + " desired"
            ax.plot(
                currentTrajectory["agentsPositionX"][:, i],
                currentTrajectory["agentsPositionYDesired"][:, i],
                "-",
                label=nameDesired,
                color=self.m_colors[i % len(self.m_colors)],
            )
            cf = ax.scatter(
                currentTrajectory["agentsPositionX"][:, i],
                currentTrajectory["agentsPositionY"][:, i],
                s=self.m_scatterSymbolSize,
                c=currentTrajectory["time"],
                marker=self.m_linestyles[i],
                label=name,
                cmap=self.m_timeColor,
            )
        # Add start points of actions
        ax = self.addStartPointOfAction(ax, currentTrajectory, "2D")
        cb = fig.colorbar(cf, ax=ax)
        if self.m_language == "german":
            cb.set_label("Zeit $[s]$", fontsize=self.m_textsize)
            ax.set_title("$x$-$y$-$t$ Fahrzeugbewegung", fontsize=self.m_textsize)
        else:
            cb.set_label("time $[s]$", fontsize=self.m_textsize)
            ax.set_title("Vehicle Movement x-y-t", fontsize=self.m_textsize)
        # Comment out if plot of road not desired
        self.drawRoad(ax)
        if plotLegend:
            ax.legend(
                bbox_to_anchor=(0, 1.02, 1, 0.2),
                loc="lower left",
                mode="expand",
                borderaxespad=0,
                ncol=currentTrajectory["numberAgents"],
            )
        ax.set_xlabel("Position x [m]", fontsize=self.m_textsize)
        ax.set_ylabel("Position y [m]", fontsize=self.m_textsize)
        ax.set_xlim(self.m_xMin, self.m_xMax)
        ax.set_ylim(self.m_yMin, self.m_yMax)  #
        ax.grid(True)

        return ax

    def drawScenario(
        self,
        ax: Axes,
        scenarioInfo: dict,
        drivenTrajectory: Optional[dict],
        annotateAgents: bool,
        displayRandomStartPos: bool = False,
    ) -> Tuple[Axes, List[dict]]:
        """
        Draws a scenario and optionally a specified trajectory.

        TODO: merge with drawTrajectory2d, drawTrajectory2dSingleAgent

        Parameters
        ----------
        ax
            the matplotlib axes
        scenarioInfo
            the scenario json object
        currentTrajectory
            trajectory data
        annotateAgents
            true, if the velocity of the agents should be drawn into the plot
        displayRandomStartPos
            if true and the scenario supports random start positions, then the normal distribution is drawn into the plot.

        Returns
        -------
        ax
            the matplotlib axes
        object_info
            list of dictionaries: `{"rectangleObject": <rectangle bounding box>, "triangleObject": <triangle, if objectType=="vehicle">}`
        """
        # static obstacles
        ax = self.drawObstacles(ax, scenarioInfo)
        # road boundaries
        ax = self.drawRoad(ax)
        # random start positions
        if displayRandomStartPos:
            ax = self.drawRandomStartPositions(ax, scenarioInfo)

        # agents
        ax, agent_plot_objects = self.drawAgents(ax, scenarioInfo, annotateAgents)
        # trajectory
        if drivenTrajectory:
            for j in range(0, drivenTrajectory["agentsPositionX"].shape[1]):
                # if agentsPredefined[j]:
                #    continue
                ax.plot(
                    drivenTrajectory["agentsPositionX"][:, j],
                    drivenTrajectory["agentsPositionY"][:, j],
                    "-.",
                    color="#0065a3",
                )

        # axis description
        ax.tick_params(labelsize=self.m_textsize - 4)
        ax.set_xlabel("x [m]", fontsize=self.m_textsize)
        ax.set_ylabel("y [m]", fontsize=self.m_textsize)
        ax.set_aspect("equal")
        ax.get_yaxis().set_label_coords(-0.03, 0.5)
        return ax, agent_plot_objects

    def drawAgents(
        self, ax: Axes, scenarioInfo: dict, annotate: bool = True
    ) -> Tuple[Axes, List[dict]]:
        """
        Draws the objects of all agents into the plot

        TODO: there is lots of duplicated code which could be replaced by calling this method...

        Parameters
        ----------
        ax
            the matplotlib axes
        scenarioInfo
            the scenario json object
        annotate
            true, if the velocity of the agents should be drawn into the plot

        Returns
        -------
        ax
            the matplotlib axes
        dict
            list of dictionaries: `{"rectangleObject": <rectangle bounding box>, "triangleObject": <triangle, if objectType=="vehicle">}`
        """
        plot_objects = []
        for index, agent in enumerate(scenarioInfo["agents"]):
            if agent["vehicle"]["position_y"] <= scenarioInfo["road"]["lane_width"]:
                yOffset = 3.5
            else:
                yOffset = -3.5
            if agent["vehicle"]["velocity_x"] > 0:
                xOffset = -7
            else:
                xOffset = 1
            color = self.m_colors[index % len(self.m_colors)]
            vehicle = agent["vehicle"]
            # Draw vehicles
            ax, plot_object = self.drawObject(
                ax,
                "vehicle",
                vehicle["position_x"],
                vehicle["position_y"],
                vehicle["heading"],
                vehicle["length"],
                vehicle["width"],
                color,
            )
            plot_objects.append(plot_object)
            if annotate:
                # Annotate agent ID
                ax.annotate(
                    "$" + str(vehicle["velocity_x"]) + " \\frac{m}{s}" + "$",
                    xy=(
                        vehicle["position_x"],
                        vehicle["position_y"],
                    ),
                    xytext=(
                        vehicle["position_x"] + xOffset,
                        vehicle["position_y"],
                    ),
                    fontsize=15,
                    bbox=dict(fc="w", ec="k"),
                )
        return ax, plot_objects

    def drawRandomStartPositions(self, ax: Axes, scenarioInfo: dict) -> Axes:
        """
        Draws the convex hull of all possible start positions into the plot.

        Parameters
        ----------
        ax
            the matplotlib axes
        scenarioInfo
            the scenario json object

        Returns
        -------
        ax
            the matplotlib axes
        """

        for id_a, points in enumerate(calculateStartPosPolygon(scenarioInfo)):
            np_points = np.array([x for y in points for x in y])
            hull = ConvexHull(np_points)
            ax.fill(
                np_points[hull.vertices, 0],
                np_points[hull.vertices, 1],
                self.m_colors[id_a],
                alpha=0.3,
            )

            ax.add_patch(Polygon(points[0], closed=True, fill=False, color="gray"))
            ax.add_patch(Polygon(points[1], closed=True, fill=False, color="gray"))
            ax.add_patch(Polygon(points[2], closed=True, fill=False, color="gray"))
            ax.add_patch(Polygon(points[3], closed=True, fill=False, color="gray"))

            # possible future extension: draw the probability as well
            # from scipy.stats import multivariate_normal
            # import matplotlib.patches as patches
            # x, y = np.mgrid[pos_x-sigma_x*2:pos_x+sigma_x*2:sigma_x/100, pos_y-sigma_y*2:pos_y+sigma_y*2:sigma_y/100]
            # pos = np.dstack((x, y))
            # rv = multivariate_normal([pos_x, pos_y], [[sigma_x, 0], [0, sigma_y]])
            # cs = ax.contourf(x, y, rv.pdf(pos), alpha=0.6, levels=40, vmax=0.1, vmin=0)
            # circ = patches.Rectangle((pos_x-sigma_x/2, pos_y-sigma_y/2), sigma_x*2, sigma_y*2, transform=ax.transData)
            # for coll in cs.collections:
            #   coll.set_clip_path(circ)
        return ax

    def setXLimits(
        self, ax: Axes, drivenTrajectory: Dict[str, Any], currentIndex: int
    ) -> Axes:
        """
        Set image view to current extract
        extract set to be 100m
        optimal visualization???


        Parameters
        ----------
        ax
            the matplotlib axes
        drivenTrajectory
            trajectory data
        currentIndex
            the current step

        Returns
        -------
        ax
            the matplotlib axes
        """
        xMax = np.max(drivenTrajectory["agentsPositionX"][currentIndex, :])
        xMin = np.min(drivenTrajectory["agentsPositionX"][currentIndex, :])
        windowWidth = 100
        # Make video a little wider than the vehicles position
        offset = 10
        if xMax - xMin < windowWidth:
            windowXMin = xMin - offset
            windowXMax = windowXMin + windowWidth
            ax.set_xlim(windowXMin, windowXMax)
        else:
            ax.set_xlim(xMin - offset, xMax + offset)

        return ax

    def drawSingleShotPlans(
        self, ax: Axes, singleShotPlans: List[Dict[str, Any]], numberStages: int
    ) -> Axes:
        """
        Draws the single shot plants

        Parameters
        ----------
        ax
            the matplotlib axes
        singleShotPlans
            trajectory data
        numberStages
            number of stages that should be drawn

        Returns
        -------
        ax
            the matplotlib axes
        """
        for i in range(0, numberStages):
            currentPlan = singleShotPlans[i]
            if currentPlan["corruptedDataSet"]:
                continue
            for j in range(0, currentPlan["numberAgents"]):
                # singleShotColor = "#D3D3D3"
                ###################
                # Use the below for varying color (a little to much for the untrained observer)
                # Varying gray scale: 220 light gray, 115 dark gray
                # rgbValue = (220 - 115*i/(numberStages-1))/255
                # singleShotColor = (rgbValue,rgbValue,rgbValue)
                rgbValue = 220 / 255
                singleShotColor = (rgbValue, rgbValue, rgbValue)
                if i == numberStages - 1:
                    rgbValue = 115 / 255
                    singleShotColor = (rgbValue, rgbValue, rgbValue)
                ax.scatter(
                    currentPlan["agentsPositionX"][:, j],
                    currentPlan["agentsPositionY"][:, j],
                    color=singleShotColor,
                    s=self.m_scatterSymbolSize - 10,
                )  # , marker = self.m_linestyles[j]
        return ax

    def drawCurrentSingleShotPlan(self, ax: Axes, currentPlan: Dict[str, Any]) -> Axes:
        """
        Draws the single shot plants

        Parameters
        ----------
        ax
            the matplotlib axes
        currentPlan
            trajectory data

        Returns
        -------
        ax
            the matplotlib axes
        """
        if currentPlan["corruptedDataSet"]:
            return ax
        for j in range(0, currentPlan["numberAgents"]):
            ax.scatter(
                currentPlan["agentsPositionX"][:, j],
                currentPlan["agentsPositionY"][:, j],
                marker=self.m_linestyles[j],
                color="#FF0000",
                s=self.m_scatterSymbolSize - 10,
            )
            ax = self.addStartPointOfAction(ax, currentPlan, "singleShot")

        return ax

    def drawEgoReward(
        self, ax: Axes, currentTrajectory: Dict[str, Any], maxEgoReward: float
    ) -> Axes:
        """
        Draws the ego reward over time

        Parameters
        ----------
        ax
            the matplotlib axes
        currentTrajectory
            trajectory data
        maxEgoReward
            maximum ego reward

        Returns
        -------
        ax
            the matplotlib axes
        """
        for i in range(0, currentTrajectory["numberAgents"]):
            if currentTrajectory["agentsPredefined"][i]:
                continue
            name = "Agent " + str(int(currentTrajectory["agentsID"][0, i]))
            ax.plot(
                currentTrajectory["time"],
                currentTrajectory["agentsEgoReward"][:, i],
                "-o",
                label=name,
                color=self.m_colors[i % len(self.m_colors)],
            )
        ax = self.annotateAccumulatedReward(ax, currentTrajectory, "egoReward")
        # Plot maximum of ego reward
        ax.plot(
            [np.min(currentTrajectory["time"]), np.max(currentTrajectory["time"])],
            [maxEgoReward, maxEgoReward],
            label="max Ego Reward",
        )
        ax.set_xlim(
            np.min(currentTrajectory["time"]), np.max(currentTrajectory["time"])
        )
        ax.legend(fontsize=self.m_textsize, loc="center right")
        ax.tick_params(labelsize=self.m_textsize - 4)
        if self.m_language == "german":
            ax.set_xlabel("Zeit $[s]$", fontsize=self.m_textsize)
            ax.set_ylabel("Ego Bewertung", fontsize=self.m_textsize)
            ax.set_title("Ego Bewertung vs. Zeit", fontsize=self.m_textsize)
        else:
            ax.set_xlabel("time $[s]$", fontsize=self.m_textsize)
            ax.set_ylabel("Ego Reward", fontsize=self.m_textsize)
            ax.set_title("Ego Reward vs. Time", fontsize=self.m_textsize)
        ax.grid(True)
        return ax

    def drawCoopReward(self, ax: Axes, currentTrajectory: Dict[str, Any]) -> Axes:
        """
        Draws the coop reward over time

        Parameters
        ----------
        ax
            the matplotlib axes
        currentTrajectory
            trajectory data

        Returns
        -------
        ax
            the matplotlib axes
        """
        for i in range(0, currentTrajectory["numberAgents"]):
            if currentTrajectory["agentsPredefined"][i]:
                continue
            name = "Agent " + str(int(currentTrajectory["agentsID"][0, i]))
            ax.plot(
                currentTrajectory["time"],
                currentTrajectory["agentsCoopReward"][:, i],
                "-o",
                label=name,
                color=self.m_colors[i % len(self.m_colors)],
            )
        ax = self.annotateAccumulatedReward(ax, currentTrajectory, "coopReward")
        ax.legend(fontsize=self.m_textsize, loc="center right")
        ax.tick_params(labelsize=self.m_textsize - 4)
        if self.m_language == "german":
            ax.set_xlabel("Zeit $[s]$", fontsize=self.m_textsize)
            ax.set_ylabel("Kooperative Belohnung", fontsize=self.m_textsize)
            ax.set_title("Kooperative Belohnung vs. Zeit", fontsize=self.m_textsize)
        else:
            ax.set_xlabel("time $[s]$", fontsize=self.m_textsize)
            ax.set_ylabel("Coop Reward", fontsize=self.m_textsize)
            ax.set_title("Coop Reward vs. Time", fontsize=self.m_textsize)
        ax.grid(True)
        return ax

    def annotateAccumulatedReward(
        self, ax: Axes, currentTrajectory: Dict[str, Any], rewardType: str
    ) -> Axes:
        """
        Annotates the accumulated reward into the ego/coop reward plots

        Parameters
        ----------
        ax
            the matplotlib axes
        currentTrajectory
            trajectory data
        rewardType
            either "egoReward" or "coopReward"

        Returns
        -------
        ax
            the matplotlib axes
        """

        def accumulateReward(agentID, currentTrajectory, rewardType):
            # initialize accumulator
            rewardAccumulator = 0.0
            # steps per stage
            nSteps = int(
                currentTrajectory["numberDataElements"]
                / currentTrajectory["numberStages"]
            )
            for i in range(0, currentTrajectory["numberStages"]):
                rewardAccumulator = (
                    rewardAccumulator + currentTrajectory[key][i * nSteps, agentID]
                )

            return rewardAccumulator

        # select reward type
        if rewardType == "egoReward":
            key = "agentsEgoReward"
        elif rewardType == "coopReward":
            key = "agentsCoopReward"
        else:
            key = "agentsEgoReward"
            print(
                "Wrong reward type selected within accumulator, defaulting to egoReward"
            )
        precision = 2
        convertCommand = "%." + str(precision) + "f"
        # Generate String with rewards
        rewardText = "Accumulative " + rewardType
        for i in range(0, currentTrajectory["numberAgents"]):
            if currentTrajectory["agentsPredefined"][i]:
                continue
            # Calculate accumulated Coop Reward
            accumulativeReward = accumulateReward(i, currentTrajectory, key)
            accumulativeReward = convertCommand % (accumulativeReward)

            rewardText = (
                rewardText + "\n" + "Agent " + str(i) + ": " + str(accumulativeReward)
            )

        yLim = ax.get_ylim()
        ax.annotate(
            rewardText,
            xy=(
                0.5 * np.max(currentTrajectory["time"]),
                yLim[0] + 0.3 * (yLim[1] - yLim[1]),
            ),
            fontsize=self.m_textsize,
        )

        return ax

    def drawDistanceX(self, ax: Axes, currentTrajectory: Dict[str, Any]) -> Axes:
        """
        Draws the driven x-positions over time

        Parameters
        ----------
        ax
            the matplotlib axes
        currentTrajectory
            trajectory data

        Returns
        -------
        ax
            the matplotlib axes
        """
        for i in range(0, currentTrajectory["numberAgents"]):
            name = "Agent: " + str(int(currentTrajectory["agentsID"][0, i]))
            ax.plot(
                currentTrajectory["time"],
                currentTrajectory["agentsPositionX"][:, i],
                "x",
                label=name,
                color=self.m_colors[i % len(self.m_colors)],
            )
        # Add start points of actions
        ax = self.addStartPointOfAction(
            ax, currentTrajectory, "time", "agentsPositionX"
        )
        # ax.legend(fontsize=self.m_textsize)
        ax.tick_params(labelsize=self.m_textsize - 4)
        if self.m_language == "german":
            ax.set_xlabel("Zeit $[s]$", fontsize=self.m_textsize)
            ax.set_ylabel("Position $x$ $[m]$", fontsize=self.m_textsize)
            ax.set_title("Longitudinale Fahrzeugbewegung", fontsize=self.m_textsize)
        else:
            ax.set_xlabel("time [s]", fontsize=self.m_textsize)
            ax.set_ylabel("x [m]", fontsize=self.m_textsize)
            ax.set_title("longitudinal vehicle movement", fontsize=self.m_textsize)
        ax.grid(True)
        return ax

    def drawDistanceY(self, ax: Axes, currentTrajectory: Dict[str, Any]) -> Axes:
        """
        Draws the driven y-positions over time

        Parameters
        ----------
        ax
            the matplotlib axes
        currentTrajectory
            trajectory data

        Returns
        -------
        ax
            the matplotlib axes
        """
        for i in range(0, currentTrajectory["numberAgents"]):
            name = "Agent: " + str(int(currentTrajectory["agentsID"][0, i]))
            ax.plot(
                currentTrajectory["time"],
                currentTrajectory["agentsPositionY"][:, i],
                "x",
                label=name,
                color=self.m_colors[i % len(self.m_colors)],
            )
        # Add start points of actions
        ax = self.addStartPointOfAction(
            ax, currentTrajectory, "time", "agentsPositionY"
        )
        # ax.legend(fontsize=self.m_textsize)
        ax.tick_params(labelsize=self.m_textsize - 4)
        if self.m_language == "german":
            ax.set_xlabel("Zeit $[s]$", fontsize=self.m_textsize)
            ax.set_ylabel("Position $y$ $[m]$", fontsize=self.m_textsize)
            ax.set_title("Laterale Fahrzeugbewegung", fontsize=self.m_textsize)
        else:
            ax.set_xlabel("time [s]", fontsize=self.m_textsize)
            ax.set_ylabel("y [m]", fontsize=self.m_textsize)
            ax.set_title("lateral vehicle movement", fontsize=self.m_textsize)
        ax.grid(True)
        return ax

    def drawVelocityX(self, ax: Axes, currentTrajectory: Dict[str, Any]) -> Axes:
        """
        Draws the x-velocities over time

        Parameters
        ----------
        ax
            the matplotlib axes
        currentTrajectory
            trajectory data

        Returns
        -------
        ax
            the matplotlib axes
        """
        for i in range(0, currentTrajectory["numberAgents"]):
            name = "Agent: " + str(int(currentTrajectory["agentsID"][0, i]))
            ax.plot(
                currentTrajectory["time"],
                currentTrajectory["agentsVelocityX"][:, i],
                "x",
                label=name,
                color=self.m_colors[i % len(self.m_colors)],
            )
            nameD = "velocity desired agent \# " + str(
                int(currentTrajectory["agentsID"][0, i])
            )
            ax.plot(
                currentTrajectory["time"],
                currentTrajectory["agentsVelocityDesiredX"][:, i],
                label="_nolegend_",
                color=self.m_colors[i % len(self.m_colors)],
            )
        # Add start points of actions
        ax = self.addStartPointOfAction(
            ax, currentTrajectory, "time", "agentsVelocityX"
        )
        ax.tick_params(labelsize=self.m_textsize - 4)
        if self.m_language == "german":
            ax.set_xlabel("Zeit $[s]$", fontsize=self.m_textsize)
            ax.set_ylabel("Geschwindigkeit $\dot{x}$ $[m/s]$", fontsize=self.m_textsize)
            ax.set_title("Geschwindigkeit $\dot{x}$", fontsize=self.m_textsize)
        else:
            ax.set_xlabel("time [s]", fontsize=self.m_textsize)
            ax.set_ylabel("$\dot{x}$ [m/s]", fontsize=self.m_textsize)
            ax.set_title("Velocity $\dot{x}$ vs. time", fontsize=self.m_textsize)
        ax.grid(True)
        return ax

    def drawVelocityY(self, ax: Axes, currentTrajectory: Dict[str, Any]) -> Axes:
        """
        Draws the y-velocities over time

        Parameters
        ----------
        ax
            the matplotlib axes
        currentTrajectory
            trajectory data

        Returns
        -------
        ax
            the matplotlib axes
        """
        for i in range(0, currentTrajectory["numberAgents"]):
            name = "Agent: " + str(int(currentTrajectory["agentsID"][0, i]))
            plt.plot(
                currentTrajectory["time"],
                currentTrajectory["agentsVelocityY"][:, i],
                "x",
                label=name,
                color=self.m_colors[i % len(self.m_colors)],
            )
        # Add start points of actions
        ax = self.addStartPointOfAction(
            ax, currentTrajectory, "time", "agentsVelocityY"
        )
        # ax.legend(fontsize=self.m_textsize)
        ax.tick_params(labelsize=self.m_textsize - 4)
        if self.m_language == "german":
            ax.set_xlabel("Zeit $[s]$", fontsize=self.m_textsize)
            ax.set_ylabel("Geschwindigkeit $\dot{y}$ $[m/s]$", fontsize=self.m_textsize)
            ax.set_title("Geschwindigkeit $\dot{y}$ vs. Zeit", fontsize=self.m_textsize)
        else:
            ax.set_xlabel("time [s]", fontsize=self.m_textsize)
            ax.set_ylabel("$\dot{y}$ [m/s]", fontsize=self.m_textsize)
            ax.set_title("Velocity $\dot{y}$ vs. time", fontsize=self.m_textsize)
        ax.grid(True)
        return ax

    def drawAccelerationX(self, ax: Axes, currentTrajectory: Dict[str, Any]) -> Axes:
        """
        Draws the x-accelerations over time

        Parameters
        ----------
        ax
            the matplotlib axes
        currentTrajectory
            trajectory data

        Returns
        -------
        ax
            the matplotlib axes
        """
        for i in range(0, currentTrajectory["numberAgents"]):
            name = "Agent: " + str(int(currentTrajectory["agentsID"][0, i]))
            plt.plot(
                currentTrajectory["time"],
                currentTrajectory["agentsAccelerationX"][:, i],
                "-",
                label=name,
                color=self.m_colors[i % len(self.m_colors)],
            )
        # Add start points of actions
        ax = self.addStartPointOfAction(
            ax, currentTrajectory, "time", "agentsAccelerationX"
        )
        # ax.legend(fontsize=self.m_textsize)
        ax.tick_params(labelsize=self.m_textsize - 4)
        if self.m_language == "german":
            ax.set_xlabel("Zeit $[s]$", fontsize=self.m_textsize)
            ax.set_ylabel("$\ddot{x}$ [m/s]", fontsize=self.m_textsize)
            ax.set_title("Beschleunigung $\ddot{x}$ vs. Zeit", fontsize=self.m_textsize)
        else:
            ax.set_xlabel("time [s]", fontsize=self.m_textsize)
            ax.set_ylabel("$\ddot{x}$ [m/s]", fontsize=self.m_textsize)
            ax.set_title("Acceleration $\ddot{x}$ vs. time", fontsize=self.m_textsize)
        ax.grid(True)
        return ax

    def drawAccelerationY(self, ax: Axes, currentTrajectory: Dict[str, Any]) -> Axes:
        """
        Draws the y-accelerations over time

        Parameters
        ----------
        ax
            the matplotlib axes
        currentTrajectory
            trajectory data

        Returns
        -------
        ax
            the matplotlib axes
        """
        for i in range(0, currentTrajectory["numberAgents"]):
            name = "Agent: " + str(int(currentTrajectory["agentsID"][0, i]))
            plt.plot(
                currentTrajectory["time"],
                currentTrajectory["agentsAccelerationY"][:, i],
                "-",
                label=name,
                color=self.m_colors[i % len(self.m_colors)],
            )
        # Add start points of actions
        ax = self.addStartPointOfAction(
            ax, currentTrajectory, "time", "agentsAccelerationY"
        )
        # ax.legend(fontsize=self.m_textsize)
        ax.tick_params(labelsize=self.m_textsize - 4)
        if self.m_language == "german":
            ax.set_xlabel("Zeit $[s]$", fontsize=self.m_textsize)
            ax.set_ylabel("$\ddot{y}$ [m/s]", fontsize=self.m_textsize)
            ax.set_title("Beschleunigung $\ddot{y}$ vs. Zeit", fontsize=self.m_textsize)
        else:
            ax.set_xlabel("time [s]", fontsize=self.m_textsize)
            ax.set_ylabel("a_y [m/s]", fontsize=self.m_textsize)
            ax.set_title("a_y vs. time", fontsize=self.m_textsize)
        ax.grid(True)
        return ax

    def drawTotalAcceleration(
        self, ax: Axes, currentTrajectory: Dict[str, Any]
    ) -> Axes:
        """
        Draws the absolute acceleration over time

        Parameters
        ----------
        ax
            the matplotlib axes
        currentTrajectory
            trajectory data

        Returns
        -------
        ax
            the matplotlib axes
        """
        for i in range(0, currentTrajectory["numberAgents"]):
            name = "agent: " + str(int(currentTrajectory["agentsID"][0, i]))
            plt.plot(
                currentTrajectory["time"],
                currentTrajectory["agentsTotalAcceleration"][:, i],
                "-",
                label=name,
                color=self.m_colors[i % len(self.m_colors)],
            )
            plt.plot(
                [np.min(currentTrajectory["time"]), np.max(currentTrajectory["time"])],
                [
                    self.m_agentsLimits["agentsMaxAbsoluteAcceleration"][i],
                    self.m_agentsLimits["agentsMaxAbsoluteAcceleration"][i],
                ],
                "r",
                label="Max Absolute Acceleration $|a_{abs,max}|$",
            )
        # Add start points of actions
        ax = self.addStartPointOfAction(
            ax, currentTrajectory, "time", "agentsTotalAcceleration"
        )
        ax.legend(fontsize=self.m_textsize)
        if self.m_language == "german":
            ax.set_xlabel("Zeit $[s]$", fontsize=self.m_textsize)
            ax.set_ylabel("$|a_{abs}| [m/s^2]$", fontsize=self.m_textsize)
            ax.set_title(
                "Absolute Beschleunigung $|a_{abs}|$ vs. Zeit", fontsize=self.m_textsize
            )
        else:
            ax.set_xlabel("time [s]", fontsize=self.m_textsize)
            ax.set_ylabel("$|a_{abs}| [m/s^2]$", fontsize=self.m_textsize)
            ax.set_title("$|a_{abs}|$ vs. time", fontsize=self.m_textsize)
        ax.grid(True)
        return ax

    def drawSteeringAngle(self, ax: Axes, currentTrajectory: Dict[str, Any]) -> Axes:
        """
        Draws the steering angle over time.

        Parameters
        ----------
        ax
            the matplotlib axes
        currentTrajectory
            trajectory data

        Returns
        -------
        ax
            the matplotlib axes
        """
        for i in range(0, currentTrajectory["numberAgents"]):
            name = "Agent: " + str(int(currentTrajectory["agentsID"][0, i]))
            plt.plot(
                currentTrajectory["time"],
                currentTrajectory["agentsSteeringAngle"][:, i],
                "-",
                label=name,
                color=self.m_colors[i % len(self.m_colors)],
            )
            plt.plot(
                [np.min(currentTrajectory["time"]), np.max(currentTrajectory["time"])],
                [
                    self.m_agentsLimits["agentsMaxSteeringAngle"][i],
                    self.m_agentsLimits["agentsMaxSteeringAngle"][i],
                ],
                "r",
                label="Max Steering Angle $\delta$",
            )
            plt.plot(
                [np.min(currentTrajectory["time"]), np.max(currentTrajectory["time"])],
                [
                    -self.m_agentsLimits["agentsMaxSteeringAngle"][i],
                    -self.m_agentsLimits["agentsMaxSteeringAngle"][i],
                ],
                "r",
                label="Min Steering Angle $\delta$",
            )
        # Add start points of actions
        ax = self.addStartPointOfAction(
            ax, currentTrajectory, "time", "agentsSteeringAngle"
        )
        ax.legend(fontsize=self.m_textsize)
        if self.m_language == "german":
            ax.set_xlabel("Zeit $[s]$", fontsize=self.m_textsize)
            ax.set_ylabel("$\delta_{veh} [deg]$", fontsize=self.m_textsize)
            ax.set_title("Lenkwinkel $\delta_{veh}$ vs. Zeit", fontsize=self.m_textsize)
        else:
            ax.set_xlabel("time [s]", fontsize=self.m_textsize)
            ax.set_ylabel("$\delta_{veh} [deg]$", fontsize=self.m_textsize)
            ax.set_title("$\delta_{veh}$ vs. time", fontsize=self.m_textsize)
        ax.grid(True)
        return ax

    def drawHeading(self, ax: Axes, currentTrajectory: Dict[str, Any]):
        """
        Draws the heading over time.

        Parameters
        ----------
        ax
            the matplotlib axes
        currentTrajectory
            trajectory data

        Returns
        -------
        ax
            the matplotlib axes
        """
        for i in range(0, currentTrajectory["numberAgents"]):
            if currentTrajectory["agentsPredefined"][i]:
                continue
            name = "Agent: " + str(int(currentTrajectory["agentsID"][0, i]))
            # Convert rad to degree
            tempHeading = np.empty(
                shape=[
                    currentTrajectory["numberDataElements"],
                    currentTrajectory["numberAgents"],
                ]
            )
            for j in range(0, currentTrajectory["numberDataElements"]):
                tempHeading[j, i] = import_proseco_data.convertRadToDegree(
                    currentTrajectory["agentsHeading"][j, i]
                )
            plt.plot(
                currentTrajectory["time"],
                tempHeading[:, i],
                "--x",
                label=name,
                color=self.m_colors[i % len(self.m_colors)],
            )
        # Add start points of actions
        currentTrajectory["tempHeading"] = tempHeading
        ax = self.addStartPointOfAction(ax, currentTrajectory, "time", "tempHeading")
        ax.legend(fontsize=self.m_textsize)
        if self.m_language == "german":
            ax.set_xlabel("Zeit $[s]$", fontsize=self.m_textsize)
            ax.set_ylabel("$\psi$ [deg]", fontsize=self.m_textsize)
            ax.set_title("Fahrzeugorientierung vs. Zeit", fontsize=self.m_textsize)
        else:
            ax.set_xlabel("time [s]", fontsize=self.m_textsize)
            ax.set_ylabel("Heading $\psi$ [deg]", fontsize=self.m_textsize)
            ax.set_title("Heading vs. time", fontsize=self.m_textsize)
        ax.grid(True)
        return ax

    def highlightCurrentState(
        self, ax: Axes, currentTrajectory: Dict[str, Any], stage: int
    ) -> Axes:
        """
        Highlights the current stage in the trajectory plot.

        Parameters
        ----------
        ax
            the matplotlib axes
        currentTrajectory
            trajectory data
        stage
            the current stage (step)

        Returns
        -------
        ax
            the matplotlib axes
        """
        currentIndex = (
            stage
            * currentTrajectory["numberDataElements"]
            / currentTrajectory["numberStages"]
        )
        for i in range(0, currentTrajectory["numberAgents"]):
            x = currentTrajectory["agentsPositionX"][int(currentIndex), i]
            y = currentTrajectory["agentsPositionY"][int(currentIndex), i]
            ax.plot(x, y, "o", color="r")
        return ax

    def annotateCurrentState(
        self,
        fig: Figure,
        currentIndex: int,
        currentTrajectory: Dict[str, Any],
        agentsPredefined: list,
    ) -> list:
        """
        Annotates the current state.

        Parameters
        ----------
        fig
            the matplotlib figure
        currentIndex
            the current stage (step)
        currentTrajectory
            trajectory data
        agentsPredefined

        Returns
        -------
        commentHandles
            the matplotlib comment objects
        """
        commentHandles = []

        numberComments = agentsPredefined.count(False)
        defaultText = "test"

        commentCount = 0
        for i in range(0, currentTrajectory["numberAgents"]):
            if agentsPredefined[i]:
                continue
            comment = plt.gcf().text(
                0.02,
                0.85 - commentCount * 0.8 / numberComments,
                defaultText,
                fontsize=14,
            )
            commentHandles.append(comment)
            commentCount = commentCount + 1

        self.updateCurrentState(
            commentHandles, currentIndex, currentTrajectory, agentsPredefined
        )
        # plt.subplots_adjust(top=0.95)

        return commentHandles

    def updateCurrentState(
        self,
        commentHandles,
        currentIndex: int,
        currentTrajectory: Dict[str, Any],
        agentsPredefined: list,
    ):
        """
        Updates the current state.

        Parameters
        ----------
        commentHandles
            the matplotlib comment objects
        currentIndex
            the current stage (step)
        currentTrajectory
            trajectory data
        agentsPredefined
        """
        precision = 2
        convertCommand = "%." + str(precision) + "f"
        commentCount = 0
        for i in range(0, currentTrajectory["numberAgents"]):
            if agentsPredefined[i]:
                continue
            xPosition = convertCommand % (
                currentTrajectory["agentsPositionX"][currentIndex, i]
            )
            xVelocity = convertCommand % (
                currentTrajectory["agentsVelocityX"][currentIndex, i]
            )
            xVelocityDesired = convertCommand % (
                currentTrajectory["agentsVelocityDesiredX"][currentIndex, i]
            )
            yPosition = convertCommand % (
                currentTrajectory["agentsPositionY"][currentIndex, i]
            )
            yPositionDesired = convertCommand % (
                currentTrajectory["agentsPositionYDesired"][currentIndex, i]
            )
            yVelocity = convertCommand % (
                currentTrajectory["agentsVelocityY"][currentIndex, i]
            )
            currentHeading = import_proseco_data.convertRadToDegree(
                currentTrajectory["agentsHeading"][currentIndex, i]
            )
            heading = convertCommand % currentHeading

            text_vehicle_state = (
                "Agent: "
                + str(i)
                + "\n"
                + "$x$: "
                + xPosition
                + " $m$"
                + "\n"
                + "$v_{x}$: "
                + xVelocity
                + " $m/s$"
                + "\n"
                + "$v_{x,Des}$: "
                + xVelocityDesired
                + " $m/s$"
                + "\n"
                + "$y$: "
                + yPosition
                + " $m$"
                + "\n"
                + "$y_{Des}$: "
                + yPositionDesired
                + " $m$"
                + "\n"
                + "$v_{y}$: "
                + yVelocity
                + " $m/s$"
                + "\n"
                + "$\\psi$: "
                + heading
                + " deg"
            )

            commentHandles[commentCount].set_text(text_vehicle_state)
            commentCount = commentCount + 1


class NodeVisualizer:
    """A class used as visualizer of node related data"""

    ### Plot Options
    ### Set the number of subplots
    rows = 1
    # cols = numberAgents

    # Textsizes etc.
    textsize = 24
    colorMap = cm.winter  # autumn_r
    symbolsize = 50
    suptitle = "Exploration of the action space; node at depth: "  # + str(depth) + ' number of child nodes: ' + str(numberDataElements)
    xlabel = "$\Delta v_{longitudinal} [m/s]$"
    ylabel = "$\Delta y_{lateral} [m]$"
    zlabelActionValue = "Action Value $Q(s,a)$"
    zlabelVisitCount = "Visit count $n_{visits}$"
    zlabelUCTscore = "$UCT_{score}$"
    zlabelPointDensity = (
        "Point density $\\frac{\# of points}{[\\frac{m}{s}] \cdot [m]}$"
    )

    ## Arrow style
    headWith = 0.3
    headLength = 0.5
    ### Plots
    figureWidth = 20
    dpiNumber = 80
    faceColor = "w"
    edgeColor = "k"
    imageFormat = 9 / 16

    ## Font size for plots
    m_textsize = 16
    m_symbolsize = 8
    m_scatterSymbolSize = 50
    m_linestyles = ["x", "o", "s", "s"]
    m_colors = ("b", "g", "r", "c", "m", "y", "k", "r")
    m_colorMap = cm.winter
    m_timeColor = cm.cool

    ## bar plot
    barOffset = 0.5
    barWidth = 1.0

    def __init__(self, textsize):
        ## Plot options
        self.m_textsize = textsize

    ## Draws the boundaries of the moves group to the action space
    def drawMoveGroups(self, ax, currentAgent, moveGroupData):
        numberMoveGroups = moveGroupData["numberDataElements"]
        minVelocityChange = moveGroupData["agentsMinVelocityChange"]
        maxVelocityChange = moveGroupData["agentsMaxVelocityChange"]
        minLateralChange = moveGroupData["agentsMinLateralChange"]
        maxLateralChange = moveGroupData["agentsMaxLateralChange"]

        for i in range(0, numberMoveGroups):
            xPoints = [
                maxVelocityChange[i, currentAgent],
                minVelocityChange[i, currentAgent],
                minVelocityChange[i, currentAgent],
                maxVelocityChange[i, currentAgent],
                maxVelocityChange[i, currentAgent],
            ]
            yPoints = [
                maxLateralChange[i, currentAgent],
                maxLateralChange[i, currentAgent],
                minLateralChange[i, currentAgent],
                minLateralChange[i, currentAgent],
                maxLateralChange[i, currentAgent],
            ]
            ax.plot(xPoints, yPoints, "b")

        return ax

    def annotateMoveGroups(self, ax, currentAgent, moveGroupData):
        numberMoveGroups = moveGroupData["numberDataElements"]
        minVelocityChange = moveGroupData["agentsMinVelocityChange"]
        maxVelocityChange = moveGroupData["agentsMaxVelocityChange"]
        minLateralChange = moveGroupData["agentsMinLateralChange"]
        maxLateralChange = moveGroupData["agentsMaxLateralChange"]
        moveGroup = moveGroupData["agentsActionClass"]

        precision = 2
        convertCommand = "%." + str(precision) + "f"

        for i in range(0, numberMoveGroups):
            # annotate Name of action
            offset = 0.25
            xText = (
                maxVelocityChange[i, currentAgent] + minVelocityChange[i, currentAgent]
            ) / 2 + offset
            yText = (
                maxLateralChange[i, currentAgent] + minLateralChange[i, currentAgent]
            ) / 2 + offset

            num_actionValue = moveGroupData["agentsActionValue"][i, currentAgent]
            num_actionVisits = moveGroupData["agentsActionVisitCount"][i, currentAgent]
            num_uctScore = moveGroupData["agentsActionUCT"][i, currentAgent]

            actionValue = convertCommand % (num_actionValue)
            actionVisits = convertCommand % (num_actionVisits)
            uctScore = convertCommand % (num_uctScore)

            currentMoveGroup = moveGroup[currentAgent][i]

            text = (
                currentMoveGroup
                + "\n"
                + "Q(s,a): "
                + actionValue
                + "\n"
                + "$N_{visits}$: "
                + actionVisits
                + "\n"
                + "UCT: "
                + uctScore
            )

            ax.text(xText, yText, text, fontsize=14)

        return ax

    def drawStatisticsBarDiagram(
        self, fig, row, maxRows, currentData, stage, dataType="Q-Values"
    ):
        if dataType == "Q-Values":
            dataKey = "agentsActionValue"
        elif dataType == "Visit Count":
            dataKey = "agentsActionVisitCount"
        elif dataType == "UCT Score":
            dataKey = "agentsActionUCT"
        else:
            print(
                "Wrong Data Type within bar diagram for Q-Values, defaulting to Q-Values"
            )
            dataKey = "agentsActionValue"
            dataType == "Q-Values"

        # store the bar objects of the different subplots
        bars = {}
        # store the axes objects of the different subplots
        barAxes = {}

        # only plot ego agent's statistic

        for i in range(0, currentData["numberAgents"]):
            ax = fig.add_subplot(
                maxRows,
                currentData["numberAgents"],
                (row - 1) * currentData["numberAgents"] + i + 1,
            )
            # prepare data for plot
            barLabels = []
            qValues = []
            barTicks = []

            for j in range(0, currentData["numberDataElements"]):
                barTicks.append(self.barOffset + j * self.barWidth)
                barLabels.append(((currentData["agentsActionClass"])[i])[j])
                qValues.append((currentData[dataKey])[j, i])

            ax, currentBars = self.drawBarDiagram(
                ax, barTicks, qValues, dataType, barLabels, i, currentData["stage"]
            )
            bars[i] = currentBars
            barAxes[i] = ax

        barplotDetails = {}
        barplotDetails["bars"] = bars
        barplotDetails["barAxes"] = barAxes

        return fig, barplotDetails

    def getQValues(self, fig, row, currentData, stage):
        # store data in dictionary
        qData = {}
        xValues = {}
        xLabels = {}
        yValues = {}

        # only plot ego agent's statistic

        for i in range(0, currentData["numberAgents"]):
            # prepare data for plot
            barLabels = []
            qValues = []
            barTicks = []

            for j in range(0, currentData["numberDataElements"]):
                barTicks.append(self.barOffset + j * self.barWidth)
                barLabels.append(((currentData["agentsActionClass"])[i])[j])
                qValues.append((currentData["agentsActionValue"])[j, i])

            # store data
            xValues[i] = barTicks
            xLabels[i] = barLabels
            yValues[i] = qValues

        qData["xValues"] = xValues
        qData["xLabels"] = xLabels
        qData["yValues"] = yValues

        return qData

    def drawBarDiagram(
        self, ax, barTicks, yValues, yCriteria, barLabels, agentID, stage
    ):
        # Plot bar diagramm and save bar objects for later manipulation
        bars = ax.bar(barTicks, yValues, color=["steelblue"])
        # Highlight maximum Q Value
        maxIndex = yValues.index(max(yValues))
        bars[maxIndex].set_color("red")
        ax.set_xticks(barTicks)
        ax.set_xticklabels(barLabels)
        ax.tick_params()
        ax.set_xlabel("Action Class")
        ax.set_ylabel(yCriteria)
        ax.set_title("Agent \#: " + str(agentID) + " , stage: " + str(stage))
        ax.grid(True)
        # Draw horizontal line for clearness
        ax.plot([0, 500], [0, 0], "k")
        ax.set_xlim(0, max(barTicks) + self.barWidth / 2)
        # ax.set_ylim()

        return ax, bars
