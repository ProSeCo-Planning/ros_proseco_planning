#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Puts the app, layout and callbacks together to create the dashboard."""

from dash import html, Input, Output, dcc

from proseco.dashboard import (
    callbacks,
    callbacks_tab_agent,
    callbacks_tab_evaluation,
    callbacks_tab_run,
    callbacks_tab_tree,
)
from proseco.dashboard.app import app
from proseco.dashboard.layouts import layout


app.layout = html.Div(
    children=[
        dcc.Location(id="url", refresh=False),
        html.Div(id="page-content"),
    ]
)

# default layout for a "404 - Not Found" page
layout_404 = html.Div(
    children=[
        html.H1("404"),
        html.H2("Not Found"),
    ],
    style={
        "text-align": "center",
        "position": "absolute",
        "top": "50%",
        "left": "50%",
        "transform": "translate(-50%, -50%)",
    },
)


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def display_page(pathname: str) -> html.Div:
    """Displays the page according to the URL pathname.

    Parameters
    ----------
    pathname : str
        URL pathname

    Returns
    -------
    html.Div
        page content
    """
    if pathname == "/":
        return layout
    else:
        return layout_404


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", dev_tools_hot_reload=False)
