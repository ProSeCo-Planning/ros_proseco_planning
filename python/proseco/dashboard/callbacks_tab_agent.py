#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Holds the callbacks that call the logic to update the agent tab."""

from typing import List
from dash import dcc, Input, Output, exceptions

from proseco.dashboard.app import app
from proseco.dashboard.app_tab_agent import generate_sample_space_plot_detailed

### RUN SAMPLE SPACE PLOT
@app.callback(
    Output("run_sample_space_plot", "children"),
    [
        Input("run_directory", "data"),
        Input("step_slider", "value"),
        Input("agent_ids", "value"),
        Input("sample_space_plot_options", "value"),
    ],
)
def update_run_sample_space_plot(
    run_directory: str, step: int, agent_ids: List[int], plot_options: str
) -> List[dcc.Graph]:
    """
    Refresh the trajectory plot of the selected run.
    Called whenever a new run, step or agent has been selected.

    Parameters
    ----------
    run_directory : str
        evaluation output directory of the selected run.
    step : int
        selected step of the trajectory.
    agent_ids : List[int]
        selected agent id.
    plot_options : str
        string indicating how the figure should be plotted. Options: "contour_visits_circle_value", "contour_value_circle_visits"

    Returns
    -------
    List[dcc.Graph]
        plotly figure for every given agent.
    """
    if not run_directory or agent_ids is None:
        raise exceptions.PreventUpdate
    if plot_options == "contour_visits_circle_value":
        contour_z = "action_visits"
        circle_size = "action_value"
    else:
        contour_z = "action_value"
        circle_size = "action_visits"
    return generate_sample_space_plot_detailed(
        run_directory, step, agent_ids, contour_z, circle_size
    )
