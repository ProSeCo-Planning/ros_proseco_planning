#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Holds the callbacks that call the logic to update the run tab."""

from typing import Any, Dict, List, Tuple, Union
import os
from pathlib import Path
from dash import Input, Output, callback_context, exceptions
import plotly.graph_objects as go

from proseco.dashboard.app import app, get_figures_and_style_dicts
from proseco.dashboard.app_tab_run import (
    generate_action_class_plot,
    generate_trajectory_plot,
)
from proseco.visualization.scenario_video_visualizer import ScenarioVideoVisualizer

### MAKE VIDEO
@app.callback(
    [
        Output("run_video", "src"),
        Output("run_video", "autoPlay"),
        Output("run_video", component_property="style"),
        Output("video_progress_container", component_property="style"),
    ],
    [
        Input("evaluation_table", "derived_virtual_data"),
        Input("evaluation_table", "derived_virtual_selected_rows"),
        Input("video_button", "n_clicks"),
    ],
)
def make_video(
    data: List[Dict[str, Any]], rows: List[int], n_clicks: int
) -> Tuple[str, bool, Dict[str, str], Dict[str, str]]:
    """
    Creates a new trajectory video (if it does not yet exist) and updates the source of the html video object.
    Called whenever a new run has been selected or the "generate video" button has been clicked.

    Parameters
    ----------
    data : List[Dict[str, Any]]
        table which contains information about all runs of the selected evaluation.
    rows : List[int]
        list of selected runs.
    n_clicks : int
        number of times the "generate video" button has been clicked.

    Returns
    -------
    Tuple[str, bool, Dict[str, str], Dict[str, str]]
        tuple containing:
            - url path of the video
            - True, if the video should be attempted to be played.
            - css style of the containing div for showing or hiding the video.
            - css style of the containing div for showing or hiding the generate video button and progress bar.
    """
    if not data or not rows:
        return "", False, {"display": "none"}, {"display": "none"}
    ctx = callback_context
    eval_folder = data[rows[0]]["path"]
    trajectory_file = next(Path(eval_folder).glob("trajectory_annotated*"), None)
    file_path = eval_folder + "/video.mp4"
    semaphore_path = eval_folder + "/semaphore.txt"
    file_parents = Path(file_path).parents
    file_path_relative = f"/static?eval={file_parents[2].name}&inum={file_parents[1].name}&run={file_parents[0].name}&file={Path(file_path).name}"

    if trajectory_file is None:
        # no trajectory file has been recorded, hide button and video container alltogether
        return "", False, {"display": "none"}, {"display": "none"}
    if os.path.exists(file_path):
        # the video already exists
        return file_path_relative, True, {"display": "block"}, {"display": "none"}
    elif ctx.triggered[0]["prop_id"].split(".")[0] == "video_button":
        visualizer = ScenarioVideoVisualizer(eval_folder)

        def _update_progress(p):
            # This is the "official" solution for updating a progress bar
            # (without using a global variable, which is bad because of shared states)
            # see: https://github.com/plotly/dash/issues/57#issuecomment-313441186,
            #      https://github.com/facultyai/dash-bootstrap-components/issues/137#issuecomment-466565639
            #
            # this solution simply writes the current progress into a text file
            # the video_progress_interval, which is handled below, will look every 0,5s
            # into this file and update the progress bar value accordingly.
            with open(semaphore_path, "w") as f:
                f.write(str(int(p * 100)))

        _update_progress(0)
        visualizer.save(file_path, updateCallback=_update_progress)

        os.remove(semaphore_path)

        return file_path_relative, True, {"display": "block"}, {"display": "none"}
    else:
        return "", False, {"display": "none"}, {"display": "block"}


@app.callback(
    [
        Output("video_progress", "value"),
        Output("video_progress", "children"),
        Output("video_progress_interval", "disabled"),
        Output("video_button", "disabled"),
    ],
    [
        Input("video_progress_interval", "n_intervals"),
        Input("evaluation_table", "derived_virtual_data"),
        Input("evaluation_table", "derived_virtual_selected_rows"),
        Input("video_button", "n_clicks"),
    ],
)
def update_video_progress(
    n: int, data: List[Dict[str, Any]], rows: List[int], n_clicks: int
) -> Tuple[float, str, bool, bool]:
    """
    Updated the progress bar while generating a trajectory video.
    Called initially when clicking the "generate video" button and then repeatedly by the video_progress_interval.

    Parameters
    ----------
    n : int
        number of times the progress interval timer has fired.
    data : List[Dict[str, Any]]
        table which contains information about all runs of the selected evaluation.
    rows : List[int]
        list of selected runs.
    n_clicks : int
        number of times the "generate video" button has been clicked.

    Returns
    -------
    Tuple[float, str, bool, bool]
        tuple containing:
            - value of the progress bar.
            - display string of the progress bar.
            - True, if the progress interval timer should be disabled. False, if it should be enabled.
            - True, if the "generate video" button should be disabled.
    """
    if not data or not rows:
        raise exceptions.PreventUpdate
    ctx = callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    eval_folder = data[rows[0]]["path"]
    semaphore_path = eval_folder + "/semaphore.txt"
    if (
        trigger == "video_progress_interval"
        and not os.path.exists(semaphore_path)
        and os.path.exists(eval_folder + "/video.mp4")
    ):
        # video creation finished, stop timer
        return 100, "100 %", True, True

    if trigger == "video_button" or trigger == "video_progress_interval":
        if os.path.exists(semaphore_path):
            with open(semaphore_path, "r") as f:
                try:
                    progress = int(f.readline())
                except ValueError:
                    # sometimes f.readline() returns ''
                    progress = 100
        else:
            # Video creation has not started yet (make_video not yet been called).
            progress = 0

        # return the current progress and enable the video_progress_interval-Timer, so that
        # this update progress function gets called repeatedly.
        return (
            progress,
            f"{progress} %" if progress >= 5 else "",
            False,
            True,
        )
    else:
        # disable the video_progress_interval-Timer and return 0 as progress.
        return 0, "0", True, False


### RUN TRAJECTORY PLOT
@app.callback(
    [
        Output("run_trajectory_plot", "figure"),
        Output("run_trajectory_plot", "style"),
    ],
    [Input("run_directory", "data")],
)
def update_run_trajectory_plot(run_directory: str) -> List[Union[go.Figure, dict]]:
    """
    Refresh the trajectory plot of the selected run.
    Called whenever a new run has been selected.

    Parameters
    ----------
    run_directory : str
        evaluation output directory of the selected run.

    Returns
    -------
    Tuple[go.Figure, dict]
        plotly figure.
        plotly figure css style dict.
    """
    if not run_directory:
        return get_figures_and_style_dicts([None])
    return get_figures_and_style_dicts([generate_trajectory_plot(run_directory)])


### RUN ACTION, UCT VALUE AND VISITS PLOT
@app.callback(
    [
        Output("run_action_values_plot", "figure"),
        Output("run_action_values_plot", "style"),
        Output("run_uct_values_plot", "figure"),
        Output("run_uct_values_plot", "style"),
        Output("run_action_visits_plot", "figure"),
        Output("run_action_visits_plot", "style"),
    ],
    [Input("run_directory", "data")],
)
def update_run_action_class_plot(
    run_directory: str,
) -> List[Union[go.Figure, dict]]:
    """
    Refresh the trajectory plot of the selected run.
    Called whenever a new run has been selected.

    Parameters
    ----------
    run_directory : str
        evaluation output directory of the selected run.

    Returns
    -------
    Tuple[go.Figure, dict, go.Figure, dict, go.Figure, dict]
        tuple containing:
            - action value figure.
            - action value figure css style dict.
            - uct value figure.
            - uct value figure css style dict.
            - action visits figure.
            - action visits figure css style dict.
    """
    if not run_directory:
        return get_figures_and_style_dicts([None, None, None])
    return get_figures_and_style_dicts(generate_action_class_plot(run_directory))
