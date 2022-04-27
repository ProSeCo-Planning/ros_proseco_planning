#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Holds the application logic for the evaluation tab."""
from typing import Literal, Tuple, List
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

from proseco.dashboard.model import get_iteration_success_dataframe


def get_mean_heatmap(result_df: pd.DataFrame, selection: List[str]) -> go.Figure:
    """Generate a mean heatmap of various outcomes for each scenario.

    - y-axis: Each scenario in the evaluation as well as a mean over all scenarios.
    - x-axis: Possible outcomes of a run [Collision, Invalid, Desire fulfilled, Success].

    Parameters
    ----------
    result_df : pd.DataFrame
        results.json summary loaded as pandas DataFrame.
    selection : List[str]
        List of result_df columns that should be used as x-axis for the plot.

    Returns
    -------
    go.Figure
        Plotly heatmap.
    """
    mean = result_df.groupby("scenario").mean().round(2)
    mean.loc["mean"] = mean.mean().round(2)
    hmap_data_mean = mean[selection]
    fig = ff.create_annotated_heatmap(
        hmap_data_mean.values,
        colorscale="Viridis_r",
        xgap=1,
        ygap=1,
        x=list(hmap_data_mean.columns),
        y=list(hmap_data_mean.index),
    )
    return fig


def get_std_heatmap(result_df: pd.DataFrame, selection: List[str]) -> go.Figure:
    """Generate a standard deviation heatmap of various outcomes for each scenario.

    - y-axis: Each scenario in the evaluation as well as a mean over all scenarios.
    - x-axis: Possible outcomes of a run [Collision, Invalid, Desire fulfilled, Success].

    Parameters
    ----------
    result_df : pd.DataFrame
        results.json summary loaded as pandas DataFrame.
    selection : List[str]
        List of result_df columns that should be used as x-axis for the plot.

    Returns
    -------
    go.Figure
        Plotly heatmap.
    """
    std = result_df.groupby("scenario").std().round(2)
    std.loc["mean"] = std.mean().round(2)
    hmap_data_std = std[selection]
    fig = ff.create_annotated_heatmap(
        hmap_data_std.values,
        colorscale="Viridis_r",
        xgap=1,
        ygap=1,
        x=list(hmap_data_std.columns),
        y=list(hmap_data_std.index),
    )
    return fig


def get_iteration_heatmap(result_df: pd.DataFrame) -> go.Figure:
    """Generates a heatmap showing the success rate for different numbers of iterations (columns) and scenarios (rows).

    Parameters
    ----------
    result_df : pd.DataFrame
        DataFrame containing the scenario, number of iterations, success rate and further values.

    Returns
    -------
    go.Figure
        plotly figure.
    """
    success_df = get_iteration_success_dataframe(result_df)
    mean_pivot = success_df.pivot(
        index="scenario", columns="n_iterations", values="mean_success_rate"
    )
    mean_pivot.loc["mean"] = mean_pivot.mean()
    mean_pivot = mean_pivot.round(2)

    fig = ff.create_annotated_heatmap(
        mean_pivot.values,
        # annotation_text=mean_text,
        x=list(mean_pivot.columns),
        y=list(mean_pivot.index),
        colorscale="Viridis_r",
        xgap=1,
        ygap=1,
    )
    # NOTE: if the x axis type is "category", the x values of the annotations must be integers starting from 0
    # WORKAROUND START
    for i, annot in enumerate(fig.layout.annotations):
        new_x = i % len(mean_pivot.columns)
        annot.x = new_x
    # WORKAROUND END
    return fig


def generate_result_heatmap(summary: pd.DataFrame) -> go.Figure:
    """Generate the result heatmap of the selected evaluation.

    The figure consits of three individual heatmaps plotting:
        1. Mean heatmap: Scenarios (rows) vs. Outcomes (columns)
        1. Standard deviation heatmap: Scenarios (rows) vs. Outcomes (columns)
        1. Iterations heatmap: Scenarios (rows) vs. MCTS Iterations (columns)

    Parameters
    ----------
    path : str
        path to the evaluation folder.

    Returns
    -------
    go.Figure
        plotly figure.
    """
    # Selection must be a list for pandas indexing
    selection = [
        "carsCollided",
        "carsInvalid",
        "desiresFulfilled",
        "success",
    ]
    fig = make_subplots(
        subplot_titles=("Mean", "Standard Deviation", "Success Rate"), rows=1, cols=3
    )

    mean_hmap = get_mean_heatmap(result_df=summary, selection=selection)
    std_hmap = get_std_heatmap(result_df=summary, selection=selection)
    iteration_hmap = get_iteration_heatmap(result_df=summary)

    fig.add_trace(mean_hmap.data[0], row=1, col=1)
    fig.add_trace(std_hmap.data[0], row=1, col=2)
    fig.add_trace(iteration_hmap.data[0], row=1, col=3)

    # WORKAROUND START (https://community.plotly.com/t/how-to-create-annotated-heatmaps-in-subplots/36686)
    new_annotations = [go.layout.Annotation(font_size=16)] * len(fig.layout.annotations)
    annot1 = list(mean_hmap.layout.annotations)
    annot2 = list(std_hmap.layout.annotations)
    annot3 = list(iteration_hmap.layout.annotations)
    for k in range(len(annot2)):
        annot2[k]["xref"] = "x2"
        annot2[k]["yref"] = "y2"

    for k in range(len(annot3)):
        annot3[k]["xref"] = "x3"
        annot3[k]["yref"] = "y3"

    # Concatenate all new annotations
    new_annotations += annot1 + annot2 + annot3
    fig.update_layout(annotations=new_annotations)
    # WORKAROUND END

    # NOTE: Formatting must be done on the figure containing the subplots, not the individual heatmaps
    fig.update_layout(
        width=1800,
        height=600,
        title="Result Overview",
        title_x=0.5,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=selection,
        ticktext=[
            "Collision",
            "Invalid",
            "Desire Fulfilled",
            "Success",
        ],
        row=1,
        col=1,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=selection,
        ticktext=[
            "Collision",
            "Invalid",
            "Desire Fulfilled",
            "Success",
        ],
        row=1,
        col=2,
    )
    fig.update_xaxes(
        title_text="Iterations",
        type="category",
        showgrid=False,
        side="bottom",
        row=1,
        col=3,
    )
    fig.update_yaxes(
        title_text="Scenarios",
        type="category",
        showgrid=False,
    )
    return fig


def generate_iteration_plots(summary: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """Generates plots showing the success rate for different numbers of iterations.

    Parameters
    ----------
    evaluation_directory : Union[Path, str]
        directory of the MCTS evaluation

    Returns
    -------
    Tuple[go.Figure, go.Figure]
        tuple of plotly figures. Contains:
            - heatmap
            - mean line plot
    """
    success_df = get_iteration_success_dataframe(summary, include_std=True)
    return (
        generate_iteration_line_plot(success_df, main_trace="mean"),
        generate_options_line_plot(summary),
    )


def generate_iteration_line_plot(
    success_df: pd.DataFrame, main_trace: Literal["mean"] = "mean"
) -> go.Figure:
    """Generates a line plot showing the success rate for different numbers of iterations.

    Parameters
    ----------
    success_df : pd.DataFrame
        DataFrame containing the scenario, number of iterations, success rate and further values
    main_trace : Literal["mean"]
        string indicating which main trace shall be displayed

    Returns
    -------
    go.Figure
        plotly figure
    """
    fig = go.Figure()
    scenarios = success_df["scenario"].unique()
    # dropdown for scenario selection
    dropdown_buttons = []
    # Add traces
    for i, scenario in enumerate(scenarios):
        scenario_df = success_df[success_df["scenario"] == scenario]

        if main_trace == "mean":
            traces = _get_mean_traces_for_iteration_line_plot(scenario_df)
        else:
            raise ValueError(f"Invalid main_trace argument: {main_trace}")

        # traces should contain main_trace, upper_trace and lower_trace
        assert len(traces) == 3
        fig.add_traces(traces)

        # Create a visibility_list indicating which traces are visible for this scenario
        visibility_list = [False, False, False] * i
        visibility_list.extend([True, True, True])
        visibility_list.extend([False, False, False] * (len(scenarios) - 1 - i))
        # Add dropdown button
        dropdown_buttons.append(
            dict(
                args=[
                    {"visible": visibility_list},
                    # indices of traces to restyle
                    # not specified because all traces should be restyled
                ],
                label=scenario,
                method="restyle",
            )
        )
    # End of scenario loop

    # initially, make the traces for scenario 1 visible
    fig.data[0].visible = True
    fig.data[1].visible = True
    fig.data[2].visible = True

    # Add annotation
    fig.update_layout(
        annotations=[
            dict(
                text="Scenario:",
                showarrow=False,
                x=0,
                y=1.1,
                xref="paper",
                yref="paper",
                align="left",
            )
        ]
    )
    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="middle",
            ),
        ]
    )
    # General layout settings
    fig.update_layout(
        title=dict(
            text="Success Rate",
            x=0.5,
        ),
        xaxis=dict(
            title_text="Iterations",
            type="category",
        ),
        yaxis=dict(
            range=[0, 1.2],
            tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            title_text="Success Rate",
        ),
    )
    return fig


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Converts a hex color string to RGB color integers.

    Parameters
    ----------
    hex_color : str
        hex color string

    Returns
    -------
    Tuple[int, int, int]
        tuple of integers describing the RGB (red, green, blue) colors
    """
    if hex_color[0] == "#":
        hex_color = hex_color[1:]
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)


def _get_mean_traces_for_iteration_line_plot(
    scenario_df: pd.DataFrame,
) -> Tuple[go.Scatter, go.Scatter, go.Scatter]:
    """Returns the traces for the mean of the success rate.

    Parameters
    ----------
    scenario_df : pd.DataFrame
        DataFrame containing the columns "n_iterations", "mean_success_rate", "std_success_rate"

    Returns
    -------
    Tuple[go.Scatter, go.Scatter, go.Scatter]
        tuple of traces. Contains:
            - main_trace: mean
            - upper_trace: upper bound using the standard deviation
            - lower_trace: lower bound using the standard deviation
    """
    # standard colors used by Plotly
    colors = px.colors.qualitative.Plotly

    # mean of success rate
    mean_trace = go.Scatter(
        name="Mean",
        x=scenario_df["n_iterations"],
        y=scenario_df["mean_success_rate"],
        # line=dict(color="rgb(0,100,80)"),
        mode="lines+markers",
        marker=dict(size=15),
        line=dict(width=4),
        # legendgroup="group",
        visible=False,
    )
    # upper std bound of success rate
    y = scenario_df["mean_success_rate"] + 2 * scenario_df["std_success_rate"]
    y = np.minimum(y, 1)

    upper_trace = go.Scatter(
        name="Upper bound",
        x=scenario_df["n_iterations"],
        y=y,
        mode="lines",
        # make the line invisible
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False,
        # legendgroup="group",
        visible=False,
    )
    # lower std bound of success rate
    y = scenario_df["mean_success_rate"] - 2 * scenario_df["std_success_rate"]
    y = np.maximum(y, 0)

    lower_trace = go.Scatter(
        name="Lower bound",
        x=scenario_df["n_iterations"],
        y=y,
        mode="lines",
        # make the line invisible
        line=dict(color="rgba(255,255,255,0)"),
        fill="tonexty",
        fillcolor=f"rgba{(*_hex_to_rgb(colors[0]), 0.3)}",
        showlegend=False,
        # legendgroup="group",
        visible=False,
    )
    return mean_trace, upper_trace, lower_trace


def generate_options_line_plot(summary_df: pd.DataFrame) -> go.Figure:
    """Generates a line plot showing the success rate for different choices of options.

    Parameters
    ----------
    success_df : pd.DataFrame
        DataFrame containing the scenario, number of iterations, success rate and further values

    Returns
    -------
    go.Figure
        plotly figure
    """
    fig = go.Figure()
    # Drop all the columns not needed for the visualization
    options_df = summary_df.drop(
        [
            "carsCollided",
            "carsInvalid",
            "desiresFulfilled",
            "finalstep",
            "maxSimTimeReached",
            "normalizedCoopRewardSum",
            "normalizedEgoRewardSum",
            "path",
            "scenario",
        ],
        axis=1,
    )
    # dropdown for scenario selection
    dropdown_buttons = []

    # Generate list for the options we want to iterate over
    options = [
        col for col in options_df.columns if col != "n_iterations" and col != "success"
    ]

    # Total number of traces in the plot (needed for toggling visibility)
    total_num_traces = sum([options_df[col].nunique() for col in options])

    # List of settings for styling the line plots
    line_styles = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"] * 2
    marker_styles = [
        "circle",
        "square",
        "diamond",
        "triangle-up",
        "triangle-down",
        "cross",
        "circle-open",
        "square-open",
        "diamond-open",
        "triangle-up-open",
        "triangle-down-open",
        "cross-open",
    ]

    # Add traces
    idx = 0
    for option in options:
        # Aggregate the
        aggregated_options = options_df.groupby(by=[option, "n_iterations"]).mean()[
            "success"
        ]
        for setting, line_style, marker_style in zip(
            aggregated_options.index.unique(level=0), line_styles, marker_styles
        ):
            fig.add_traces(
                go.Scatter(
                    name=str(setting),
                    x=aggregated_options.loc[setting].index,
                    y=aggregated_options.loc[setting],
                    # line=dict(color="rgb(0,100,80)"),
                    mode="lines+markers",
                    marker=dict(size=12),
                    line=dict(width=3, dash=line_style),
                    marker_symbol=marker_style,
                    visible=False,
                )
            )

        # Create a visibility_list indicating which traces are visible for this setting
        num_traces = len(aggregated_options.index.unique(level=0))
        visibility_list = np.array([False] * total_num_traces)
        visibility_list[idx : idx + num_traces] = True

        # Increment index
        idx += num_traces

        # Add dropdown button
        dropdown_buttons.append(
            dict(
                args=[
                    {"visible": visibility_list},
                    # indices of traces to restyle
                    # not specified because all traces should be restyled
                ],
                label=option,
                method="restyle",
            )
        )

    # Add annotation
    fig.update_layout(
        annotations=[
            dict(
                text="Option:",
                showarrow=False,
                x=0,
                y=1.1,
                xref="paper",
                yref="paper",
                align="left",
            )
        ]
    )
    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="middle",
            ),
        ]
    )
    # General layout settings
    fig.update_layout(
        title=dict(
            text="Success Rate",
            x=0.5,
        ),
        xaxis=dict(
            title_text="Iterations",
            type="category",
        ),
        yaxis=dict(
            range=[0, 1.2],
            tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            title_text="Success Rate",
        ),
    )
    return fig
