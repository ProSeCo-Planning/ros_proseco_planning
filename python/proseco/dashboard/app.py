#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Holds the application logic."""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from dash import Dash
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import os
import pandas as pd
import flask
from flask import request
from proseco.utility.io import get_user_temp_dir
from werkzeug.exceptions import NotFound

from proseco.dashboard.model import (
    load_result,
    load_scenario,
    get_summary,
)


def get_agent_ids(run_directory: str) -> List[str]:
    """
    Returns a list of agent ids for a specific run.

    Parameters
    ----------
    run_directory : str
        directory of the mcts evaluator output.

    Returns
    -------
    List[str]
        List of agent ids.
    """
    scenario = load_scenario(run_directory)
    return [agent["id"] for agent in scenario["agents"]]


def get_steps(run_directory: str) -> int:
    """
    Returns the max number of steps.

    Parameters
    ----------
    run_directory : str
        directory of the mcts evaluator output.

    Returns
    -------
    int
        max number of steps.
    """
    result = load_result(run_directory)
    return result["finalstep"]


def table_type(df_column: pd.Series) -> str:
    """
    Returns the data type of a pandas series for a plotly table.

    See https://dash.plotly.com/datatable/filtering

    Parameters
    ----------
    df_column
        the column of a data frame

    Returns
    -------
    str
        data type of the column
    """
    if isinstance(df_column.dtype, pd.DatetimeTZDtype):
        return "datetime"
    elif (
        isinstance(df_column.dtype, pd.StringDtype)
        or isinstance(df_column.dtype, pd.BooleanDtype)
        or isinstance(df_column.dtype, pd.CategoricalDtype)
        or isinstance(df_column.dtype, pd.PeriodDtype)
    ):
        return "text"
    elif (
        isinstance(df_column.dtype, pd.SparseDtype)
        or isinstance(df_column.dtype, pd.IntervalDtype)
        or isinstance(df_column.dtype, pd.Int8Dtype)
        or isinstance(df_column.dtype, pd.Int16Dtype)
        or isinstance(df_column.dtype, pd.Int32Dtype)
        or isinstance(df_column.dtype, pd.Int64Dtype)
    ):
        return "numeric"
    else:
        return "any"


def generate_evaluation_table(
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
    summary = get_summary(evaluation_directory)
    return (
        summary.to_dict("records"),
        [
            {
                "name": i,
                "id": i,
                "type": table_type(summary[i]),
                "format": Format(scheme=Scheme.fixed, precision=2),
            }
            for i in summary.columns
        ],
        "single",
    )


def get_figures_and_style_dicts(
    figures: Iterable[Optional[go.Figure]],
) -> List[Union[go.Figure, dict]]:
    """
    Converts a list of optional figures into an alternating sequence of figure data and css style dicts.
    If a figure is `None`, then `display` is set to `none`.

    Parameters
    ----------
    figures
        list of optional figures.

    Returns
    -------
    List[Union[go.Figure, dict]]
        sequence of alternating figures and style dicts.
    """
    res = []
    for figure in figures:
        if figure:
            res.append(figure)
            res.append({"display": "block"})
        else:
            res.append({})
            res.append({"display": "none"})
    return res


# root evaluation directory: proseco_evaluator_output
root_evaluation_directory = get_user_temp_dir() / "proseco_evaluator_output"

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
server = app.server


@server.route("/static/")
def serve_static():
    """
    Serves generated videos, which are located in the tmp evaluation directory.
    """
    eval = request.args["eval"].replace("/", "")
    inum = request.args["inum"].replace("/", "")
    run = request.args["run"].replace("/", "")
    file = request.args["file"]
    output_directory = os.path.join(root_evaluation_directory, eval, inum, run)
    if not os.path.exists(os.path.join(output_directory, file)):
        raise NotFound()
    return flask.send_from_directory(output_directory, file)
