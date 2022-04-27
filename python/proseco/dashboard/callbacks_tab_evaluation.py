#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Holds the callbacks that call the logic to update the evaluation tab."""

from typing import Any, Dict, List, Union
from dash import Input, Output
import pandas as pd
import plotly.graph_objects as go

from proseco.dashboard.app import app, get_figures_and_style_dicts
from proseco.dashboard.model import get_summary
from proseco.dashboard.app_tab_evaluation import (
    generate_iteration_plots,
    generate_result_heatmap,
)


@app.callback(
    [
        Output("evaluation_result_heatmap", "figure"),
        Output("evaluation_result_heatmap", "style"),
        Output("evaluation_scenario_line_plot_mean", "figure"),
        Output("evaluation_scenario_line_plot_mean", "style"),
        Output("evaluation_option_line_plot_mean", "figure"),
        Output("evaluation_option_line_plot_mean", "style"),
    ],
    Input("evaluation_table", "derived_virtual_data"),
    Input("evaluation_table", "derived_virtual_selected_rows"),
    [Input("evaluation_directories", "value")],
)
def update_evaluation_plots(
    rows: List[Dict[str, Any]],
    derived_virtual_selected_rows: List[Dict[str, Any]],
    evaluation_directory: str,
) -> List[Union[go.Figure, dict]]:
    """Update the evaluation tab plots.

    Generates the following figures:
        1. 3 Matrix plots with mean results, result standard deviations and iteration success per scenario.
        2. Line plot visualizing the success rate for a scenario with 1-sigma error bands.
        3. Line plot visualizing the effect of different option choices.

    An update of the plots is triggered when:
        - A new evaluation directory is selected.
        - The evaluation table is filtered.

    Parameters
    ----------
    rows : List[Dict[str, Any]]
        Rows of the evaluation table.
    derived_virtual_selected_rows : List[Dict[str, Any]]
        Selected rows after filtering and sorting across all pages.
    evaluation_directory : str
        evaluation output directory of the selected run.

    Returns
    -------
    List[Union[go.Figure, dict]]
        List containing plotly figures and css style dictionaries.
    """
    # When the table is first rendered, `derived_virtual_data` and
    # `derived_virtual_selected_rows` will be `None`. This is due to an
    # idiosyncrasy in Dash (unsupplied properties are always None and Dash
    # calls the dependent callbacks when the component is first rendered).
    # So, if `rows` is `None`, then the component was just rendered
    # and its value will be the same as the component's dataframe.
    # Instead of setting `None` in here, you could also set
    # `derived_virtual_data=df.to_rows('dict')` when you initialize
    # the component.
    # See https://dash.plotly.com/datatable/interactivity

    # Don't return anything if evaluation_directory has not been selected
    if not evaluation_directory:
        return get_figures_and_style_dicts([None, None, None])

    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []

    # Load all data if no rows are selected and only the selected rows if they are given
    dff = get_summary(evaluation_directory) if len(rows) == 0 else pd.DataFrame(rows)

    # Create single list with all figures and styles in the evaluation tab
    updated_figs = get_figures_and_style_dicts([generate_result_heatmap(dff)])
    updated_figs += get_figures_and_style_dicts(generate_iteration_plots(dff))
    return updated_figs
