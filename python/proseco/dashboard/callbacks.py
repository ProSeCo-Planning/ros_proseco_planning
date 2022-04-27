#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Holds the callbacks that call the logic to update the dashboard."""

from typing import Any, Dict, List, Tuple
from dash import html, Input, Output, exceptions
import os

from proseco.dashboard.app import (
    app,
    generate_evaluation_table,
    get_agent_ids,
    get_steps,
    root_evaluation_directory,
)
from proseco.dashboard.layouts import tab_evaluation, tab_run, tab_agent, tab_tree
from proseco.utility.io import get_list_of_directories

### RENDER CALLBACK
@app.callback(Output("tabs-content", "children"), [Input("tabs", "value")])
def render_content(tab: str) -> html.Div:
    """
    Renders the content of the selected tab.
    Called whenever a new tab has been selected.

    Parameters
    ----------
    tab : str
        name of the selected tab.

    Returns
    -------
    html.Div
        content of the tab.
    """
    if tab == "evaluation":
        return tab_evaluation
    elif tab == "run":
        return tab_run
    elif tab == "agent":
        return tab_agent
    elif tab == "tree":
        return tab_tree


### UPDATE EVALUATION DIRECTORIES
@app.callback(
    Output("evaluation_directories", "options"),
    [Input("refresh_button", "n_clicks")],
)
def update_evaluation_directories(n_clicks: int) -> List[Dict[str, str]]:
    """
    Refresh the list of directories of all evaluations.

    Parameters
    ----------
    n_clicks : int
        number of times the refresh button has been clicked.

    Returns
    -------
    List[Dict[str, str]]
        list of subdirectories (=evaluations)
    """

    directories = get_list_of_directories(root_evaluation_directory, descending=True)
    return [
        {"label": str(directory), "value": str(directory)}
        for directory in directories
        if directory.joinpath("results.json").exists()
    ]


### UPDATE RUN DIRECTORY
@app.callback(
    Output("run_directory", "data"),
    [
        Input("evaluation_table", "derived_virtual_data"),
        Input("evaluation_table", "derived_virtual_selected_rows"),
    ],
)
def update_run_directory(data: List[Dict[str, Any]], rows: List[int]) -> str:
    """
    Set the run directory to the chosen the list of directories of all runs of a specific evaluation.
    Called whenever a new run has been selected.

    Parameters
    ----------
    data : List[Dict[str, Any]]
        table which contains information about all runs of the selected evaluation.
    rows : List[int]
        list of selected runs.

    Returns
    -------
    str
        evaluation output directory of the selected run.
    """
    if not rows:
        raise exceptions.PreventUpdate
    return data[rows[0]]["path"]


### UPDATE AGENT IDS
@app.callback(Output("agent_ids", "options"), [Input("run_directory", "data")])
def update_agent_ids(run_directory: str) -> List[Dict[str, str]]:
    """
    Refresh the list of agents a specific evaluation run.
    Called whenever a new run has been selected.

    Parameters
    ----------
    run_directory : str
        evaluation output directory of the selected run.

    Returns
    -------
    List[Dict[str, str]]
        list of agent ids.
    """
    if not run_directory:
        raise exceptions.PreventUpdate
    agent_ids = get_agent_ids(run_directory)
    return [{"label": agent_id, "value": agent_id} for agent_id in agent_ids]


### UPDATE STEPS
@app.callback(
    [Output("step_slider", "max"), Output("step_slider", "marks")],
    [Input("run_directory", "data")],
)
def update_step_slider(run_directory: str) -> Tuple[int, Dict[int, str]]:
    """
    Refresh the step slider (max value and ticks).
    Called whenever a new run has been selected.

    Parameters
    ----------
    run_directory : str
        evaluation output directory of the selected run.

    Returns
    -------
    Tuple[int, Dict[int, str]]
        tuple containing:
            - maximum number of steps.
            - "ticks" for the step slider.
    """
    steps = {}
    if not run_directory:
        raise exceptions.PreventUpdate
    max_steps = get_steps(run_directory)
    for step in range(max_steps):
        steps[step] = str(step)
    return max_steps, steps


### UPDATE EVALUATION TABLE
@app.callback(
    Output("evaluation_table", "data"),
    Output("evaluation_table", "columns"),
    Output("evaluation_table", "row_selectable"),
    [Input("evaluation_directories", "value")],
)
def update_evaluation_table(
    evaluation_directory: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    """
    Generate a table from the currently selected results.json summary file.

    Parameters
    ----------
    evaluation_directory : str
        directory of the mcts evaluation (containing all performed runs).

    Returns
    -------
    Tuple[Dict[str, Any], List[Dict[str, Any]], str]
        tuple containing:
            - data for the evaluation table.
            - columns for the evaluation table.
            - "single" or "multiple", depending on how many rows should be selectable.
    """
    if not evaluation_directory:
        raise exceptions.PreventUpdate
    return generate_evaluation_table(evaluation_directory)
