#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Holds the actual html that describes the dashboard."""


from dash import html, dash_table, dcc
import dash_bootstrap_components as dbc


### MAIN LAYOUT

layout = html.Div(
    # container
    children=[
        html.Div(
            # row
            children=[
                html.Div(
                    # column
                    children=[
                        html.Div(
                            children=[
                                # html.H1("ProSeCo Dashboard"),
                                html.Label("Evaluation"),
                                html.Button(
                                    "Refresh",
                                    id="refresh_button",
                                    className="btn btn-primary m-1",
                                ),
                                dcc.Dropdown(
                                    id="evaluation_directories",
                                    placeholder="Select an evaluation",
                                ),
                            ]
                        ),
                        html.Label("Runs"),
                        dcc.Store(id="run_directory"),
                        dash_table.DataTable(
                            id="evaluation_table",
                            filter_action="native",
                            sort_action="native",
                            page_action="native",
                            page_current=0,
                            page_size=10,
                            # ---------------------
                            sort_mode="multi",
                            column_selectable="single",
                            selected_columns=[],
                            selected_rows=[],
                            # -----------------------------
                            style_cell={
                                "overflow": "hidden",
                                "textOverflow": "ellipsis",
                                "maxWidth": 0,
                            },
                        ),
                        html.Label("Agent Id"),
                        dcc.Dropdown(
                            id="agent_ids", multi=True, placeholder="Select agents"
                        ),
                        html.Label("Step"),
                        dcc.Slider(id="step_slider", min=0, max=1, step=1, value=0),
                    ],
                    className="col-md-12",
                )
            ],
            className="row",
        ),
        html.Div(
            # row
            children=[
                html.Div(
                    # column
                    children=[
                        dcc.Tabs(
                            id="tabs",
                            value="evaluation",
                            children=[
                                dcc.Tab(label="Evaluation", value="evaluation"),
                                dcc.Tab(label="Run", value="run"),
                                dcc.Tab(label="Agent", value="agent"),
                                dcc.Tab(label="Tree", value="tree"),
                            ],
                        ),
                        html.Div(id="tabs-content"),
                    ],
                    className="col-md-12",
                )
            ],
            className="row",
        ),
    ],
    className="container-fluid",
)

### TAB LAYOUTS

tab_evaluation = html.Div(
    children=[
        dcc.Loading(
            id="heatmap_loading",
            type="circle",
            children=dcc.Graph(
                id="evaluation_result_heatmap", figure={}, style={"display": "none"}
            ),
        ),
        dcc.Graph(
            id="evaluation_scenario_line_plot_mean",
            figure={},
            className="col-md",
            style={"display": "none"},
        ),
        dcc.Graph(
            id="evaluation_option_line_plot_mean",
            figure={},
            className="col-md",
            style={"display": "none"},
        ),
    ],
)

tab_run = html.Div(
    [
        html.Div(id="run_info"),
        html.Div(
            children=[
                html.Video(
                    id="run_video",
                    controls=True,
                    width=970,
                    style={"margin": "15px", "display": "none"},
                ),
                html.Br(),
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        html.Button(
                                            "Generate Video",
                                            id="video_button",
                                            className="btn btn-primary",
                                        )
                                    ],
                                    className="col-md-auto",
                                ),
                                dcc.Interval(
                                    id="video_progress_interval",
                                    n_intervals=0,
                                    interval=500,
                                    disabled=True,
                                ),
                                html.Div(
                                    children=[
                                        dbc.Progress(
                                            id="video_progress",
                                            style={"height": "2rem"},
                                        )
                                    ],
                                    className="col",
                                ),
                            ],
                            className="row align-items-center h-100",
                        ),
                    ],
                    id="video_progress_container",
                    className="container",
                    style={"display": "none"},
                ),
            ],
            id="video_content",
            className="mx-auto",
            style={"width": "1000px"},
        ),
        dcc.Loading(
            id="trajectory_loading",
            type="circle",
            children=[
                dcc.Graph(
                    id="run_trajectory_plot", figure={}, style={"display": "none"}
                ),
                dcc.Graph(
                    id="run_action_values_plot", figure={}, style={"display": "none"}
                ),
                dcc.Graph(
                    id="run_action_visits_plot", figure={}, style={"display": "none"}
                ),
                dcc.Graph(
                    id="run_uct_values_plot", figure={}, style={"display": "none"}
                ),
            ],
        ),
    ]
)

tab_agent = html.Div(
    children=[
        html.Div(
            children=[
                html.Label(
                    children=[
                        "Plot options",
                        dcc.Dropdown(
                            id="sample_space_plot_options",
                            options=[
                                {
                                    "label": "Contour: Action value; Circle: Action visit count",
                                    "value": "contour_value_circle_visits",
                                },
                                {
                                    "label": "Contour: Action visit count; Circle: Action value",
                                    "value": "contour_visits_circle_value",
                                },
                            ],
                            value="contour_value_circle_visits",
                            clearable=False,
                        ),
                    ],
                    className="col-md",
                ),
            ],
            className="row",
        ),
        html.Div(id="run_sample_space_plot", className="row"),
    ]
)

tab_tree = html.Div(
    children=[
        dcc.Store(id="tree_data"),
        html.Div(id="tree_container", children=[html.Div(id="tree-container")]),
        html.Div(id="dummy_output", hidden=True),
    ],
    className="row",
)
