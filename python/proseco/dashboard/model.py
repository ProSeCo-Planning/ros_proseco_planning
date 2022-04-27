#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Data Handling"""

from typing import Any, Dict, Generator, List, Optional, Union, cast
import pandas as pd
from pathlib import Path

from proseco.utility.io import load_data


def _convert_to_path(path: Union[Path, str]) -> Path:
    """Converts the path parameter to a Path object"""
    if isinstance(path, str):
        path = Path(path).resolve()
    return path


def load_summary(evaluation_directory: Union[Path, str]) -> List[Dict[str, Any]]:
    """Returns the result summary as a list of dictionaries.

    Parameters
    ----------
    evaluation_directory : Union[Path, str]
        directory of the MCTS evaluation

    Returns
    -------
    List[Dict[str, Any]]
        list containing the result summary for each run
    """
    evaluation_directory = _convert_to_path(evaluation_directory)
    # assumes that there exists only one file for the summary
    summary_path = list(evaluation_directory.glob("results.*"))[0]
    summary = cast(List[Dict[str, Any]], load_data(summary_path))
    return summary


def get_summary(evaluation_directory: Union[Path, str]) -> pd.DataFrame:
    """Returns the result summary as a list of dictionaries.

    Parameters
    ----------
    evaluation_directory : Union[Path, str]
        directory of the MCTS evaluation

    Returns
    -------
    List[Dict[str, Any]]
        list containing the result summary for each run
    """
    evaluation_directory = _convert_to_path(evaluation_directory)
    # NOTE: Summary is always saved as .json file
    summary = pd.read_json(evaluation_directory / "results.json")
    summary.drop(
        columns=[
            "options_scenario_uuid",
            "options_uuid",
            "options_hash",
            "options_scenario_hash",
        ],
        inplace=True,
    )
    # backwards compatibility with evaluations before terminal state has been removed
    if "terminalState" in summary.columns:
        summary.drop("terminalState", axis=1, inplace=True)
    return summary.round(2)


def get_run_directories(
    evaluation_directory: Union[Path, str]
) -> Generator[Path, None, None]:
    """Returns the run directory paths that are within an evaluation directory.

    Parameters
    ----------
    evaluation_directory : Union[Path, str]
        directory of the MCTS evaluation

    Yields
    -------
    Path
        path of a run directory
    """
    evaluation_directory = _convert_to_path(evaluation_directory)
    # summary = load_summary(evaluation_directory)
    # run_directories = [Path(run["path"]) for run in summary]
    # return run_directories

    for inum_dir in evaluation_directory.iterdir():
        if not inum_dir.is_dir():
            continue
        for run_dir in inum_dir.iterdir():
            if not run_dir.is_dir():
                continue
            yield run_dir


def load_result(run_directory: Union[Path, str]) -> Dict[str, Any]:
    """Returns the result as a dictionary.

    Parameters
    ----------
    run_directory : Union[Path, str]
        directory of the MCTS evaluator output for a specific run

    Returns
    -------
    Dict[str, Any]
        dictionary containing the result
    """
    run_directory = _convert_to_path(run_directory)
    # assumes that there exists only one file for the result
    result_path = list(run_directory.glob("result.*"))[0]
    result = cast(Dict[str, Any], load_data(result_path))
    return result


def is_successful_result(result_data: Dict[str, Any]) -> bool:
    """Checks whether a result of a specific run is successful.

    Parameters
    ----------
    result_data : Dict[str, Any]
        dictionary containing the result data of a specific run

    Returns
    -------
    bool
        True if the result is successful, False otherwise
    """
    return not (result_data["carsCollided"] or result_data["carsInvalid"])


def load_options(run_directory: Union[Path, str]) -> Dict[str, Any]:
    """Returns the options as a dictionary.

    Parameters
    ----------
    run_directory : Union[Path, str]
        directory of the MCTS evaluator output for a specific run

    Returns
    -------
    Dict[str, Any]
        dictionary containing the options
    """
    run_directory = _convert_to_path(run_directory)
    # assumes that there exists only one file for the options
    options_path = list(run_directory.glob("options_output.*"))[0]
    options = cast(Dict[str, Any], load_data(options_path))
    return options


def load_scenario(run_directory: Union[Path, str]) -> Dict[str, Any]:
    """Returns the scenario as a dictionary.

    Parameters
    ----------
    run_directory : Union[Path, str]
        directory of the MCTS evaluator output for a specific run

    Returns
    -------
    Dict[str, Any]
        dictionary containing the scenario
    """
    run_directory = _convert_to_path(run_directory)
    # assumes that there exists only one file for the scenario
    scenario_path = list(run_directory.glob("scenario_output.*"))[0]
    scenario = cast(Dict[str, Any], load_data(scenario_path))
    return scenario


def load_uuid(run_directory: Union[Path, str]) -> Dict[str, Any]:
    """Returns the UUID of the options and the options-scenario-tuple as a dictionary.

    Parameters
    ----------
    run_directory : Union[Path, str]
        directory of the MCTS evaluator output for a specific run

    Returns
    -------
    Dict[str, Any]
        dictionary containing the UUIDs
    """
    run_directory = _convert_to_path(run_directory)
    # assumes that there exists only one file for the uuid
    uuid_path = list(run_directory.glob("uuid.*"))[0]
    uuid = cast(Dict[str, Any], load_data(uuid_path))
    return uuid


def load_trajectory(run_directory: Union[Path, str]) -> Optional[Dict[str, Any]]:
    """Returns the trajectory as a dictionary.

    Parameters
    ----------
    run_directory : Union[Path, str]
        directory of the MCTS evaluator output for a specific run

    Returns
    -------
    Dict[str, Any]
        dictionary containing the trajectory
    """
    run_directory = _convert_to_path(run_directory)
    # assumes that there exists only one file for the trajectory
    try:
        trajectory_path = list(run_directory.glob("trajectory_annotated.*"))[0]
    except IndexError:
        return None
    trajectory = cast(Dict[str, Any], load_data(trajectory_path))
    return trajectory


def get_trajectory_dataframe(
    trajectory_data: Dict[str, Any], max_level: Optional[int] = None
) -> pd.DataFrame:
    """Returns the trajectory as a DataFrame.

    Parameters
    ----------
    trajectory_data : Dict[str, Any]
        dictionary containing the trajectory data
    max_level : Optional[int], optional
        max number of levels (depth of dict) to normalize.
        if None, normalizes all levels, by default None

    Returns
    -------
    pd.DataFrame
        DataFrame for the trajectory data
    """
    trajectory_df = pd.json_normalize(
        trajectory_data["agents"], ["trajectory"], "id", max_level=max_level
    )
    return trajectory_df


def load_step(
    run_directory: Union[Path, str], step: int = 0
) -> Optional[Dict[str, Any]]:
    """Returns the step data as a dictionary.

    Parameters
    ----------
    run_directory : Union[Path, str]
        directory of the MCTS evaluator output for a specific run
    step : int, optional
        step of the trajectory, by default 0

    Returns
    -------
    Dict[str, Any]
        dictionary containing the step data
    """
    run_directory = _convert_to_path(run_directory)
    # assumes that there exists only one file for this step
    try:
        step_path = list(run_directory.glob("root_node_" + str(step) + ".*"))[0]
    except IndexError:
        return None
    step_data = cast(Dict[str, Any], load_data(step_path))
    # currently, the step value isn't included in the file, so add it manually
    step_data["step"] = step
    return step_data


def get_step_dataframe(step_data: Dict[str, Any]) -> pd.DataFrame:
    """Returns the step data as a DataFrame

    Parameters
    ----------
    step_data : Dict[str, Any]
        dictionary containing the step data

    Returns
    -------
    pd.DataFrame
        DataFrame for the step data
    """
    step_df = pd.json_normalize(step_data["agents"], ["actions"], "id")

    step_df["step"] = step_data["step"]
    return step_df


def get_step_from_file_name(step_path: Path) -> int:
    """Extracts the step from the file name.

    Parameters
    ----------
    step_path : Path
        path to the `root_node_{step}.*` file

    Returns
    -------
    int
        step of the trajectory
    """
    file_name = step_path.name
    file_prefix = "root_node_"
    start = file_name.index(file_prefix) + len(file_prefix)
    end = file_name.rindex(".")
    step = int(file_name[start:end])
    return step


def get_iteration_success_dataframe(
    iteration_df: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    """Returns a DataFrame containing the success rate of "scenario"-"number of iterations" combinations.

    Parameters
    ----------
    iteration_df : pd.DataFrame
        DataFrame containing the result data, a success indicator and the number of iterations
    **kwargs
        keyword arguments for the function `get_feature_success_dataframe`

    Returns
    -------
    pd.DataFrame
        DataFrame containing the scenario, number of iterations, success rate and further values specified by the parameters
    """
    # if n_iterations was not iterated over, we need to add it to the dataframe
    if "n_iterations" not in iteration_df:
        options_output = load_data(Path(iteration_df.path[0]) / "options_output.json")
        iteration_df["n_iterations"] = options_output["compute_options"]["n_iterations"]
    return get_feature_success_dataframe("n_iterations", iteration_df, **kwargs)


def get_feature_success_dataframe(
    feature: str,
    result_df: pd.DataFrame,
    *,
    include_std: bool = False,
) -> pd.DataFrame:
    """Returns a DataFrame containing the success rate of "scenario"-"feature" combinations.

    Parameters
    ----------
    feature: str
        the feature that must be present in `result_df`
    result_df : pd.DataFrame
        DataFrame containing the result data, a success indicator and the feature
    include_std : bool, optional
        True if the standard deviation should be included, by default False

    Returns
    -------
    pd.DataFrame
        DataFrame containing the scenario, the feature, the success rate and further values specified by the parameters
    """
    rate_grouped = result_df.groupby([feature, "scenario"])[["success"]]
    success_df = rate_grouped.mean().round(2)
    if success_df["success"].dtype == "bool":
        success_df["success"] = success_df["success"].astype("float")

    success_df.rename(columns={"success": "mean_success_rate"}, inplace=True)

    # list of DataFrames that will later be merged with the success_df
    df_list = []

    if include_std:
        std_df = rate_grouped.std().round(2)
        std_df.rename(columns={"success": "std_success_rate"}, inplace=True)
        df_list.append(std_df)

    if df_list:
        success_df = success_df.join(df_list)

    success_df.reset_index(inplace=True)
    return success_df
