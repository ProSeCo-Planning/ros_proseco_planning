#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Holds the callbacks that call the logic to update the agent tab."""

from typing import Any, Dict, List
from dash import Input, Output, html
from pathlib import Path
import glob

from proseco.dashboard.app import app


@app.callback(
    Output("tree_data", "data"),
    Output("tree_container", "children"),
    Input("evaluation_table", "derived_virtual_data"),
    Input("evaluation_table", "derived_virtual_selected_rows"),
    Input("step_slider", "value"),
)
def create_file_path(data: List[Dict[str, Any]], rows: List[int], step: int):
    """Creates the file path for the search tree visualization.

    Args:
        data List[Dict[str, Any]]: The table that contains information about all runs of the selected evaluation.
        rows List[int]: The list of selected runs.
        step int: The currently selected step of the run.

    Returns:
        Dict[str, Any]: A dictionary that stores the file path to the search tree.
    """
    if not data or not rows:
        return {"data": False, "path": ""}, []
    eval_folder = data[rows[0]]["path"]

    file_path = (
        glob.glob(f"{eval_folder}/search_tree_*")[0]
        if glob.glob(f"{eval_folder}/search_tree_*")
        else ""
    )

    if file_path and file_path.endswith(".msgpack"):
        file_path = f"{eval_folder}/search_tree_{step}.msgpack"
    elif file_path and file_path.endswith(".json"):
        file_path = f"{eval_folder}/search_tree_{step}.json"
    else:
        return {"data": False, "path": ""}, []

    file_parents = Path(file_path).parents
    file_path_relative = f"/static?eval={file_parents[2].name}&inum={file_parents[1].name}&run={file_parents[0].name}&file={Path(file_path).name}"
    return {"data": True, "path": file_path_relative}, html.Div(id="tree-container")


# Callback for the JavaScript function
app.clientside_callback(
    """
    function(tree_data) {
        if(tree_data.data === true) {
            loadJSONandRender(tree_data.path);
        }
        else {
            alert("No tree data found. Enable tree export for visualization.")
        }
    }
    """,
    Output("dummy_output", "value"),
    Input("tree_data", "data"),
)
