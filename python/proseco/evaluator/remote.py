import ray
import uuid as uu_id
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Tuple, List, Any

from proseco.utility.io import get_list_of_files, load_data

from proseco.evaluator.util import (
    serialize_dict_for_cli,
    validate_result,
)


@ray.remote(num_returns=1)
def run(
    options: Dict[str, Dict[str, Any]],
    scenario: Dict[str, Any],
    uuids: Dict[str, str],
    binary_path: str,
    debug: bool,
    pba: ray.actor.ActorHandle,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Run the evaluation.
    Evaluates the given task, specified by a scenario and an options dictionary
    and returns the results to the head node.
    Notes
    -----
    The result files are deleted after they have been loaded.
    Parameters
    ----------
    options : Dict[str, Dict[str, Any]]
        Dictionary containing the options for the evaluation.
    scenario: Dict[str, Any]
        Dictionary containing the scenario for the evaluation.
    uuid: Dict[str, str]
        Dictionary containing the unique IDs of the options and the scenario.
    info_dict: Dict[str, Any]
        Dictionary containing information about the progress of the evaluation.
    Returns
    -------
    List[Any]
        Results of the evaluation.
    info_dict: Dict[str, Any]
        Information to update the progress bar.
    """
    # Create a temporary directory for the evaluation
    temp_directory = create_temporary_evaluation_directory()
    options["output_options"]["output_path"] = str(temp_directory)
    # Serialize the options and scenario so they can be passed via the command line
    serialized_options = serialize_dict_for_cli(options)
    serialized_scenario = serialize_dict_for_cli(scenario)
    node_name = generate_node_name()
    # Run the evaluation
    dump_core = "ulimit -c unlimited"
    run_cmd = f"{binary_path}/proseco_planning_node {node_name} {serialized_options} {serialized_scenario}"
    if debug:
        try:
            subprocess.run(
                f"{dump_core} && {run_cmd}",
                shell=True,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"{e.stderr} \nORIGINATING FROM\n{e.cmd}")
            raise e
    else:
        try:
            subprocess.run(run_cmd, shell=True, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"{e.stderr} \nORIGINATING FROM\n{e.cmd}")
            raise e
    results = fetch_results(temp_directory, uuids)
    # Remove the temporary directory as well as its containing files
    shutil.rmtree(temp_directory)
    pba.update.remote(1)
    return results


def fetch_results(path: str, uuid: Dict[str, str]) -> List[Any]:
    """Get the evaluation results from the temporary file.
    Fetches all the output files of the worker-task and returns them in a list.
    Inside the lists are either lists like such : [<file_name>, <opened_file>]
    where <opened_file> is a dictionary for .json files.
    Parameters
    ----------
    path: Path
        pathlib Path to the result files.
    uuid: Dict[str, str]
        Dictionary containing the unique IDs of the options and the scenario.
    Returns
    -------
    List[Any]
        Result list as loaded from the file.
    """
    results: List[Any] = []
    for file_path in get_list_of_files(path):
        data = load_data(file_path)
        if file_path.name == "result.json":
            assert validate_result(data)
        results.append([file_path.name, data])
    results.append(["uuid.json", uuid])
    return results


def create_temporary_evaluation_directory() -> Path:
    """Creates a temporary evaluation directory.
    Returns
    -------
    Path
        Path to the temporary evaluation directory.
    """
    path = Path("/tmp") / f"proseco_evaluation_{uu_id.uuid4()}"
    path.mkdir(exist_ok=False)
    return path


def generate_node_name() -> str:
    """Generates a node name using a uuid.
    Returns
    -------
    str
        The generated node name.
    """
    return str(uu_id.uuid4()).replace("-", "")
