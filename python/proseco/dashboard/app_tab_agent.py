#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Holds the application logic for the agent tab."""

from typing import List
from dash import dcc
import plotly.express as px
import plotly.graph_objects as go

from proseco.dashboard.model import (
    get_step_dataframe,
    load_step,
)


def generate_sample_space_plot(
    run_directory: str, step: int = 0, agent_ids=None
) -> List[dcc.Graph]:
    """
    Generates the sample space plot of the given agents.

    Parameters
    ----------
    run_directory : str
        directory of the mcts evaluator output.
    step : int
        the currently selected step.
    agent_ids : List[str]
        list of agent ids for each of which a plot should be generated.

    Returns
    -------
    List[dcc.Graph]
        plotly figure for every given agent.
    """
    figs: List[dcc.Graph] = []

    if not agent_ids:
        # list agent_ids is empty, so return an empty figure list
        return figs

    step_data = load_step(run_directory, step)
    df = get_step_dataframe(step_data)

    for agent_id in agent_ids:
        x = df[df["id"] == agent_id]["d_velocity"]
        y = df[df["id"] == agent_id]["d_lateral"]

        chosen_x = df[(df["id"] == agent_id) & df["action_chosen"]]["d_velocity"]
        chosen_y = df[(df["id"] == agent_id) & df["action_chosen"]]["d_lateral"]

        fig = go.Figure()
        fig.add_trace(
            go.Histogram2dContour(
                x=x,
                y=y,
                colorscale="Plasma",
                xaxis="x",
                yaxis="y",
                nbinsx=5,
                nbinsy=5,
                hoverinfo="none",
                colorbar={"title": "Frequency"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                xaxis="x",
                yaxis="y",
                mode="markers",
                name="Sampled Action",
                marker=dict(color="black", size=5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=chosen_x,
                y=chosen_y,
                xaxis="x",
                yaxis="y",
                mode="markers",
                name="Selected Action",
                marker=dict(
                    line=dict(width=2, color="DarkSlateGrey"),
                    color="red",
                    size=15,
                    symbol="x",
                ),
            )
        )

        fig.add_trace(
            go.Histogram(
                y=y,
                xaxis="x2",
                name="Sampling hist",
                marker=dict(color="rgba(0,0,0,1)"),
            )
        )
        fig.add_trace(
            go.Histogram(
                x=x,
                yaxis="y2",
                name="Sampling hist",
                marker=dict(color="rgba(0,0,0,1)"),
            )
        )

        fig.update_layout(
            # autosize = False,
            title=f"Agent: {agent_id}",
            title_x=0.5,
            xaxis=dict(zeroline=False, domain=[0, 0.85], showgrid=False),
            yaxis=dict(zeroline=False, domain=[0, 0.85], showgrid=False),
            xaxis2=dict(zeroline=False, domain=[0.85, 1], showgrid=False),
            yaxis2=dict(zeroline=False, domain=[0.85, 1], showgrid=False),
            # height = 600,
            # width = 600,
            bargap=0,
            hovermode="closest",
            showlegend=False,
            xaxis_title="Velocity change",
            yaxis_title="Lateral change",
        )
        figs.append(fig)
    return [
        dcc.Graph(figure=fig, className="col-sm-12 col-md-6 col-lg-4") for fig in figs
    ]


def generate_sample_space_plot_circles(
    run_directory: str, step: int = 0, agent_ids=None
) -> List[dcc.Graph]:
    """
    Generates the sample space plot of the given agents.

    Parameters
    ----------
    run_directory : str
        directory of the mcts evaluator output.
    step : int
        the currently selected step.
    agent_ids : List[str]
        list of agent ids for each of which a plot should be generated.

    Returns
    -------
    List[dcc.Graph]
        plotly figure for every given agent.

    """
    figs: List[dcc.Graph] = []

    if step_data := load_step(run_directory, step) is None:
        return [
            dcc.Graph(
                figure=go.Figure(
                    {
                        "layout": {
                            "xaxis": {"visible": False},
                            "yaxis": {"visible": False},
                            "annotations": [
                                {
                                    "text": "Enable childMap export for sample space visualization.",
                                    "xref": "paper",
                                    "yref": "paper",
                                    "showarrow": False,
                                    "font": {"size": 28},
                                }
                            ],
                        }
                    }
                )
            )
        ]
    elif not agent_ids:
        # list agent_ids is empty, so return an empty figure list
        return figs

    df = get_step_dataframe(step_data)

    for agent_id in agent_ids:
        # finally chosen action
        chosen_x = df[(df["id"] == agent_id) & df["action_chosen"]]["d_velocity"]
        chosen_y = df[(df["id"] == agent_id) & df["action_chosen"]]["d_lateral"]

        fig = px.density_contour(
            df[df["id"] == agent_id],
            x="d_velocity",
            y="d_lateral",
            nbinsx=7,
            nbinsy=7,
            marginal_x="histogram",
            marginal_y="histogram",
            color_discrete_sequence=px.colors.sequential.Plasma,
            labels={"d_velocity": "Velocity change", "d_lateral": "Lateral change"},
        )
        fig.data[0]["contours_coloring"] = "fill"
        fig.add_trace(
            px.scatter(
                df[df["id"] == agent_id],
                x="d_velocity",
                y="d_lateral",
                hover_data=["action_visits", "action_value"],
                labels={
                    "d_velocity": "Velocity change",
                    "d_lateral": "Lateral change",
                    "action_visits": "Action visit count",
                    "action_value": "Action value",
                },
                size="action_visits",
            )
            .update_traces(
                marker=dict(
                    line=dict(width=1, color="black"),
                    opacity=0.5,
                    symbol="circle-dot",
                    color="grey",
                )
            )
            .data[0]
        )

        fig.add_trace(
            go.Scatter(
                x=chosen_x,
                y=chosen_y,
                xaxis="x",
                yaxis="y",
                mode="markers",
                name="Selected Action",
                marker=dict(
                    line=dict(width=2, color="DarkSlateGrey"),
                    color="red",
                    size=15,
                    symbol="x",
                ),
            )
        )

        fig.update_layout(
            title=f"Agent: {agent_id}, Step: {step}",
            title_x=0.5,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        figs.append(fig)
    return [
        dcc.Graph(figure=fig, className="col-sm-12 col-md-6 col-lg-4") for fig in figs
    ]


def generate_sample_space_plot_detailed(
    run_directory: str,
    step: int = 0,
    agent_ids: List[int] = [0],
    contour_z: str = "action_value",
    circle_size: str = "action_visits",
) -> List[dcc.Graph]:
    """Generates detailed sample space plots for the given agents.

    Parameters
    ----------
    run_directory : str
        directory of the mcts evaluator output.
    step : int, optional
        the currently selected step, by default 0
    agent_ids : List[int], optional
        list of agent ids for each of which a plot should be generated, by default [0]
    contour_z : str, optional
        string indicating which dataframe column the contour should display, by default "action_value"
    circle_size : str, optional
        string indicating which dataframe column determines the circle size, by default "action_visits"

    Returns
    -------
    List[dcc.Graph]
        plotly figure for every given agent.
    """
    figs: List[dcc.Graph] = []

    step_data = load_step(run_directory, step)
    if step_data is None:
        return [
            dcc.Graph(
                figure=go.Figure(
                    {
                        "layout": {
                            "xaxis": {"visible": False},
                            "yaxis": {"visible": False},
                            "annotations": [
                                {
                                    "text": "Enable childMap export for sample space visualization.",
                                    "xref": "paper",
                                    "yref": "paper",
                                    "showarrow": False,
                                    "font": {"size": 28},
                                }
                            ],
                        }
                    }
                )
            )
        ]
    elif not agent_ids:
        # list agent_ids is empty, so return an empty figure list
        return figs

    df = get_step_dataframe(step_data)
    # this variable must be set here because the variable circle_size may change
    use_action_value_circles = circle_size == "action_value"

    for agent_id in agent_ids:
        agent_df = df[df["id"] == agent_id]

        # drop unvisited actions
        agent_df = agent_df[agent_df["action_visits"] != 0]

        # finally chosen action
        chosen_x = agent_df[agent_df["action_chosen"] > 0]["d_velocity"]
        chosen_y = agent_df[agent_df["action_chosen"] > 0]["d_lateral"]

        labels = {
            "d_velocity": "Velocity change",
            "d_lateral": "Lateral change",
            "action_visits": "Action visit count",
            "action_value": "Action value",
        }

        if use_action_value_circles:
            # action values can be negative, so transform them to positive values for the circle size
            min_value = agent_df["action_value"].min()
            if min_value < 0:
                circle_size = agent_df["action_value"] - min_value

        fig = px.scatter(
            agent_df,
            x="d_velocity",
            y="d_lateral",
            marginal_x="histogram",
            marginal_y="histogram",
            hover_data=["action_visits", "action_value"],
            labels=labels,
            size=circle_size,
        ).update_traces(
            marker=dict(
                line=dict(width=1, color="black"),
                opacity=0.5,
                symbol="circle-dot",
                color="grey",
            ),
            selector=dict(type="scatter"),
        )

        pivot_df = agent_df.pivot(
            index="d_lateral", columns="d_velocity", values=contour_z
        )

        fig.add_trace(
            go.Contour(
                z=pivot_df.values,
                x=pivot_df.columns.values,
                y=pivot_df.index.values,
                contours_coloring="heatmap",  # "fill"
                connectgaps=True,
                # line_smoothing=1.3,
                colorscale=px.colors.sequential.Plasma,
                xaxis="x",
                yaxis="y",
                hoverinfo="skip",
                colorbar=dict(title=labels[contour_z], titleside="right"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=chosen_x,
                y=chosen_y,
                xaxis="x",
                yaxis="y",
                mode="markers",
                name="Selected Action",
                marker=dict(
                    line=dict(width=2, color="DarkSlateGrey"),
                    color="red",
                    size=15,
                    symbol="x",
                ),
            )
        )

        # determine min/max x/y values to specify the axes ranges manually
        min_x = agent_df.loc[:, "d_velocity"].min()
        max_x = agent_df.loc[:, "d_velocity"].max()
        min_y = agent_df.loc[:, "d_lateral"].min()
        max_y = agent_df.loc[:, "d_lateral"].max()

        fig.update_layout(
            title=dict(
                text=f"Agent: {agent_id}, Step: {step}",
                x=0.5,
            ),
            margin_t=110,  # default: 100
            height=460,  # default: 450
            xaxis_range=[min_x, max_x],
            yaxis_range=[min_y, max_y],
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        figs.append(fig)
    return [
        dcc.Graph(figure=fig, className="col-sm-12 col-md-6 col-lg-4") for fig in figs
    ]
