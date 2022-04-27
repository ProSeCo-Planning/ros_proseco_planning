#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Holds the application logic for the run tab."""

from typing import Any, Dict, List, Tuple, Union, cast, Optional
import math
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from proseco.dashboard.model import (
    get_step_dataframe,
    get_step_from_file_name,
    get_trajectory_dataframe,
    load_data,
    load_trajectory,
)


def rotate_coordinates(
    coordinates: List[np.ndarray],
    anchor: np.ndarray = np.array([0, 0]),
    angle: float = 0.0,
) -> List[np.ndarray]:
    """Rotates x-y-coordinates around an anchor for a given angle.

    Parameters
    ----------
    coordinates : List[np.ndarray]
        list of x-y-coordinates
    anchor : np.ndarray, optional
        x-y-coordinates of the anchor around which is rotated, by default np.array([0, 0])
    angle : float, optional
        angle to rotate, measured in radians, by default 0.0

    Returns
    -------
    List[np.ndarray]
        list of the rotated x-y-coordinates
    """
    if angle == 0.0:
        return coordinates

    # center the coordinates around the origin like they were centered around the anchor
    coordinates_centered = [c - anchor for c in coordinates]
    # rotate around the origin
    rotation_matrix = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    new_coordinates_centered = [rotation_matrix @ c for c in coordinates_centered]
    # retranslate the coordinates
    new_coordinates = [c + anchor for c in new_coordinates_centered]
    return new_coordinates


def get_object_vertices(
    obj: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns the vertices of an object

    Parameters
    ----------
    obj : Dict[str, Any]
        dictionary containing the position, width, length and heading of the object

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        x-y-coordinates of the vertices front_right, front_left, back_left, back_right
    """
    x_pos = obj["position_x"]
    y_pos = obj["position_y"]
    width = obj["width"]
    length = obj["length"]
    coordinates = [
        np.array([x_pos + length, y_pos - width / 2]),  # front_right
        np.array([x_pos + length, y_pos + width / 2]),  # front_left
        np.array([x_pos, y_pos + width / 2]),  # back_left
        np.array([x_pos, y_pos - width / 2]),  # back_right
    ]
    vertices = rotate_coordinates(
        coordinates, anchor=np.array([x_pos, y_pos]), angle=obj["heading"]
    )
    return (
        vertices[0],  # front_right
        vertices[1],  # front_left
        vertices[2],  # back_left
        vertices[3],  # back_right
    )


def get_heading_arrow(obj: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the vertices of an arrow indicating the heading of an object.

    Parameters
    ----------
    obj : Dict[str, Any]
        dictionary containing the position, width, length and heading of the object

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        x-y-coordinates of the vertices tip, left, right
    """
    x_pos = obj["position_x"]
    y_pos = obj["position_y"]
    width = obj["width"]
    length = obj["length"]
    # offset so that the arrow has an angle of ~ 45°
    offset = length - 0.707107 * width  # 0.707107 ~= math.sqrt(0.5)
    coordinates = [
        np.array([x_pos + length, y_pos]),  # tip
        np.array([x_pos + offset, y_pos + width / 2]),  # left
        np.array([x_pos + offset, y_pos - width / 2]),  # right
    ]
    heading_arrow = rotate_coordinates(
        coordinates,
        anchor=np.array([x_pos, y_pos]),
        angle=obj["heading"],
    )
    return (
        heading_arrow[0],  # tip
        heading_arrow[1],  # left
        heading_arrow[2],  # right
    )


def get_object_traces(object: Dict[str, Any]) -> Tuple[go.Scatter, go.Scatter]:
    """Returns traces that indicate the vertices and the heading of an object.

    Parameters
    ----------
    object : Dict[str, Any]
        dictionary containing the position, width, length and heading of the object

    Returns
    -------
    Tuple[go.Scatter, go.Scatter]
        a trace for the vertices and a trace for the heading of the object
    """
    ver = get_object_vertices(object)
    x = [v[0] for v in ver]
    x.append(ver[0][0])
    y = [v[1] for v in ver]
    y.append(ver[0][1])
    vertices = go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line_color="black",
        showlegend=False,
    )
    tip, left, right = get_heading_arrow(object)
    heading_arrow = go.Scatter(
        x=[tip[0], left[0], right[0], tip[0]],
        y=[tip[1], left[1], right[1], tip[1]],
        mode="lines",
        line_color="black",
        showlegend=False,
        hoverinfo="skip",
    )
    return vertices, heading_arrow


def get_vehicle_traces_for_tick(
    tick: int, trajectory_df: pd.DataFrame, trajectory_json: Dict[str, Any]
) -> List[Tuple[int, go.Scatter, go.Scatter]]:
    """Returns traces for each vehicle for a specific tick of the trajectory.

    Parameters
    ----------
    tick : int
        tick of the trajectory
    trajectory_df : pd.DataFrame
        data frame containing the trajectory information
    trajectory_json : Dict[str, Any]
        dictionary representing the trajectory file

    Returns
    -------
    List[Tuple[int, go.Scatter, go.Scatter]]
        list of tuples consisting of the agent id, a trace for the vertices and a trace for the heading arrow
    """
    traces = []
    tick_df = trajectory_df[trajectory_df["tick"] == tick]
    for agent in tick_df.itertuples(index=False):
        agent_dict = dict(
            position_x=agent.position_x,
            position_y=agent.position_y,
            heading=agent.heading,
            width=trajectory_json["agents"][agent.id]["vehicle"]["width"],
            length=trajectory_json["agents"][agent.id]["vehicle"]["length"],
        )
        vertices, heading_arrow = get_object_traces(agent_dict)
        vertices.name = f"Vehicle {agent.id}"
        traces.append((agent.id, vertices, heading_arrow))
    return traces


def generate_trajectory_plot(run_directory: str) -> go.Figure:
    """
    Generates the trajectory plot for the selected run.

    Parameters
    ----------
    run_directory : str
        directory of the mcts evaluator output.

    Returns
    -------
    Optional[go.Figure]
        plotly figure.
    """
    trajectory_data = load_trajectory(run_directory)
    if trajectory_data is None:
        return go.Figure(
            {
                "layout": {
                    "xaxis": {"visible": False},
                    "yaxis": {"visible": False},
                    "annotations": [
                        {
                            "text": "No trajectory data found. Enable trajectory export for visualization.",
                            "xref": "paper",
                            "yref": "paper",
                            "showarrow": False,
                            "font": {"size": 28},
                        }
                    ],
                }
            }
        )
    # use max_level=0 because nested elements such as the action info are not relevant for this plot
    trajectory_df = get_trajectory_dataframe(trajectory_data, max_level=0)

    # Build road lanes
    lanes = []
    lanes.append(
        dict(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=0,
            xref="paper",
            line=dict(color="Black", width=2, dash="dash"),
        )
    )
    lane_width = trajectory_data["road"]["lane_width"]
    for i in range(trajectory_data["road"]["number_lanes"]):
        lanes.append(
            dict(
                type="rect",
                x0=0,
                y0=i * lane_width,
                x1=1,
                y1=(i + 1) * lane_width,
                xref="paper",
                fillcolor="LightGrey" if i % 2 == 0 else "grey",
                opacity=0.3,
                layer="below",
                line_width=0,
            )
        )
        lanes.append(
            dict(
                type="line",
                x0=0,
                y0=(i + 1) * lane_width,
                x1=1,
                y1=(i + 1) * lane_width,
                xref="paper",
                line=dict(
                    color="Black",
                    width=2,
                    dash="dash",
                ),
            )
        )

    fig = px.scatter(
        trajectory_df,
        x="position_x",
        y="position_y",
        color="tick",
        color_continuous_scale=px.colors.sequential.Plasma,
        labels={
            "id": "Agent",
            "position_x": "Longitudinal distance",
            "position_y": "Lateral distance",
            "tick": "Time Interval",
        },
    )
    fig.update_layout(shapes=lanes)

    fig.add_trace(
        go.Scatter(
            x=trajectory_df[trajectory_df["tick"] == 0]["position_x"],
            y=trajectory_df[trajectory_df["tick"] == 0]["position_y"],
            name="Starting point",
            line_color="red",
            mode="markers",
            showlegend=False,
        )
    )

    if trajectory_data["obstacles"]:
        for obstacle in trajectory_data["obstacles"]:
            vertices, heading_arrow = get_object_traces(obstacle)
            vertices.fill = "toself"
            vertices.name = f"Obstacle {obstacle['id']}"
            fig.add_trace(vertices)
            fig.add_trace(heading_arrow)

    # number of vehicles
    n_vehicles = len(trajectory_data["agents"])
    # number of traces that are added so far
    n_traces = len(fig.data)
    # ticks per trajectory step
    ticks_per_step = trajectory_df.loc[trajectory_df["step"] == 0, "tick"].max() + 1
    # last tick of the trajectory
    last_tick = trajectory_df["tick"].max()
    # steps for the slider
    steps = []

    for tick in range(last_tick + 1):
        # create the vehicle traces for each tick
        vehicle_traces = get_vehicle_traces_for_tick(
            tick, trajectory_df, trajectory_data
        )
        for _, vertices, heading_arrow in vehicle_traces:
            # make the traces invisible
            vertices.visible = False
            heading_arrow.visible = False

            fig.add_trace(vertices)
            fig.add_trace(heading_arrow)

        # visibility_list indicates which traces are visible for this tick
        visibility_list = [False, False] * (tick * n_vehicles)
        visibility_list.extend([True, True] * n_vehicles)
        visibility_list.extend([False, False] * ((last_tick - tick) * n_vehicles))

        # create a slider step for each tick
        step = dict(
            method="restyle",
            args=[
                {"visible": visibility_list},
                # indices of traces to restyle
                list(range(n_traces, n_traces + 2 * n_vehicles * (last_tick + 1))),
            ],
            label=f"step {tick // ticks_per_step}, tick {tick}",
            # value=f"{tick}",
        )
        steps.append(step)

    # initially, show traces for the last tick
    for i in range(n_vehicles):
        fig.data[-i * 2 - 1].visible = True
        fig.data[-i * 2 - 2].visible = True

    sliders = [
        dict(
            active=last_tick,
            steps=steps,
        )
    ]
    fig.update_layout(sliders=sliders)

    return fig


def generate_action_class_plot(
    run_directory: str,
) -> Tuple[Optional[go.Figure], Optional[go.Figure], Optional[go.Figure]]:
    """
    Generates plots for the specified run.

    Parameters
    ----------
    run_directory : str
        directory of the mcts evaluator output.

    Returns
    -------
    Tuple[Optional[go.Figure], Optional[go.Figure], Optional[go.Figure]]
        tuple containing:
            - action value figure.
            - uct value figure.
            - action visits figure.
    """
    steps = []
    for step_path in Path(run_directory).glob("root_node_*"):
        step_data = cast(Dict[str, Any], load_data(step_path))
        # currently, the step value isn't included in the file, so add it manually
        step_data["step"] = get_step_from_file_name(step_path)
        step_df = get_step_dataframe(step_data)
        steps.append(step_df)

    if len(steps) == 0:
        return None, None, None

    steps_df = pd.concat(steps, axis=0, ignore_index=True, copy=False)
    steps_df.sort_values(
        ["step", "id", "action_class"], inplace=True, ignore_index=True
    )

    action_value_fig = px.box(
        steps_df,
        x="action_class",
        y="action_value",
        color="id",
        animation_frame="step",
        animation_group="action_class",
        template="ggplot2",
        labels={
            "id": "Agent",
            "action_class": "Action Class",
            "action_value": "Action Value",
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    action_value_fig.update_traces(quartilemethod="linear")
    action_value_fig.update_layout(
        title_text="Action Values", title_x=0.5
    )  # or "inclusive", or "linear" by default
    action_value_fig["layout"].pop("updatemenus")

    # a little bit hacky
    if sliders := action_value_fig.layout.sliders:
        sliders[0].x = 0
        sliders[0].len = 1

    uct_value_fig = px.box(
        steps_df,
        x="action_class",
        y="action_uct",
        color="id",
        animation_frame="step",
        animation_group="action_class",
        template="ggplot2",
        labels={
            "id": "Agent",
            "action_class": "Action Class",
            "action_uct": "UCT",
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    uct_value_fig.update_traces(quartilemethod="linear")
    uct_value_fig.update_layout(
        title_text="UCT Values", title_x=0.5
    )  # or "inclusive", or "linear" by default
    uct_value_fig["layout"].pop("updatemenus")

    # a little bit hacky
    if sliders := uct_value_fig.layout.sliders:
        sliders[0].x = 0
        sliders[0].len = 1

    action_visits_fig = px.bar(
        steps_df,
        x="action_class",
        y="action_visits",
        color="id",
        animation_frame="step",
        animation_group="action_class",
        barmode="group",
        template="ggplot2",
        labels={
            "id": "Agent",
            "action_class": "Action Class",
            "action_visits": "Visits",
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    action_visits_fig.update_layout(
        title_text="Visit Counts", title_x=0.5
    )  # or "inclusive", or "linear" by default
    action_visits_fig["layout"].pop("updatemenus")

    # a little bit hacky
    if sliders := action_visits_fig.layout.sliders:
        sliders[0].x = 0
        sliders[0].len = 1

    return action_value_fig, uct_value_fig, action_visits_fig
