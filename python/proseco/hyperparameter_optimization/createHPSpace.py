"""Create a HP_space json file from a ranges and a default file

This tool accepts json files for both the ranges and defaults.

This script requires that `os`, `json`, `ros`, `argparse`, `logging` and `
typing` is installed within the Python environment you are running this script 
in.

This file can also be imported as a module and contains the following
function:

* create_hp_space: saves a hp_space in the hyperparameter optimization config
"""

from pathlib import Path
import argparse
from proseco.evaluator.util import flatten_dictionary
from proseco.utility.io import (
    get_absolute_path,
    load_data,
    save_data,
    get_absolute_path,
)
from proseco.utility.ui import get_logger

logger = get_logger("ProSeCo HP Space Generator")
from typing import Dict, List, Any


def _add_conditions(hyperparameters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Returns a list of conditions in a format that is compatible with the json
    reader of the ConfigSpace API.

    The conditions are used to activate and deactivate different groups of
    hyperparameters.

    The list is created from a list of hyperparameters. In our case, all the
    hyperparameters that (de)activate a group of hyperparameters must be named
    "active".

    Parameters
    ----------
    hyperparameters: List[Dict[str, Any]]
        List of hyperparameters

    Returns
    -------
    List
        A list of conditions
    """

    # initialize the list of conditions that will be returned later
    conditions_list = []

    # iterate over hyperparameters named XX-active and are therefore parents
    for parent_hp in hyperparameters:
        if "active" in parent_hp["name"]:

            # get the group of hyperparameters the hyperparameter activates
            group_name = parent_hp["path"].split("/")[-2]

            # iterate over every hyperparameter and check if it is in the
            # group of the parent_hp or if it activates a subgroup
            for child_hp in hyperparameters:

                in_group = group_name == child_hp["path"].split("/")[-2]
                activates_sub_group = (
                    group_name in child_hp["path"] and "active" in child_hp["name"]
                )
                if (in_group or activates_sub_group) and child_hp != parent_hp:

                    # create a condition as a dict and add it to the list
                    # of conditions
                    conditions_list.append(
                        {
                            "type": "EQ",
                            "child": child_hp["name"],
                            "parent": parent_hp["name"],
                            "value": True,
                        }
                    )

    return conditions_list


def _add_defaults(
    hyperparameter_list: List[Dict[str, Any]], defaults_dict: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Adds the default values of the hyperparameters to the list of hyperparameters

    Parameters
    ----------
    hyperparameter_list: List[Dict[str, Any]]
        A List of hyperparameters with empty default values
    defaults_dict: Dict[str, Any]
        A dictionary with the default values for each hyperparameter

    Returns
    -------
    List
        A list of hyperparameters
    """
    # setting the default value for every hyperparameter from the default_dict
    for hp in range(len(hyperparameter_list)):
        hyperparameter_list[hp]["default"] = defaults_dict[
            hyperparameter_list[hp]["path"]
        ]

    return hyperparameter_list


def _add_hyperparameters(
    ranges_path: Path, defaults_path: Path
) -> List[Dict[str, Any]]:
    """Returns a list of hyperparameters in a format that is compatible with the json
    reader of the ConfigSpace API.

    The list is created from two files: a hp_space file that defines the ranges of the
    hyperparameters and an options file that defines the default values of the
    hyperparameters. Both are in json format.

    Parameters
    ----------
    ranges_path: Path
        Path to the hp_space file
    defaults_path: Path
        Path to the options file

    Returns
    -------
    List
        A list of hyperparameters
    """

    # load the ranges of the hyperparameters as a dict
    ranges_dict = load_data(ranges_path)
    ranges_dict = flatten_dictionary(ranges_dict)

    # load the default values of the hyperparameters as a dict
    defaults_dict = load_data(defaults_path)
    defaults_dict = flatten_dictionary(defaults_dict)

    hyperparameter_list = _add_ranges(ranges_dict)

    hyperparameter_list = _add_defaults(hyperparameter_list, defaults_dict)

    return hyperparameter_list


def _add_ranges(ranges_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Creates a list of hyperparameters from a dictionary that contains the ranges
    of each hyperparameter.

    Parameters
    ----------
    ranges_dict: Dict[str, Any]
        A dictionary with the ranges/possible values for each hyperparameter

    Returns
    -------
    List
        A list of hyperparameters without the default values
    """
    # initialize the list of hyperparameters that will be returned later
    hyperparameter_list = []

    # initialize a counter. It is necessary so that the order of the
    # hyperparameters is not changed when the list is ordered.
    counter = 0

    # making the info of the hp_space/ranges dict compatible with the ConfigSpace json
    # reader for every hyperparameter and loading it into the hyperparameter_list
    for key, value in ranges_dict.items():
        counter += 1
        tmp_dict = {"path": key, "name": f"{counter:02d}-" + key.split("/")[-1]}

        # setting the possible values for categorical/boolean hyperparameters or the
        # lower and upper bounds for float/integer hyperparameters, respectively
        tmp_dict = _set_type_and_bounds(tmp_dict, value)

        # appending the dict of the current hyperparameter to the list of
        # hyperparameters
        hyperparameter_list.append(tmp_dict)

    return hyperparameter_list


def _set_type_and_bounds(tmp_dict: Dict[str, Any], value: Any) -> Dict[str, Any]:
    """Returns a dict that can be added to a list of hyperparameters.

    The dict extends the tmp_dict with information from value.

    Parameters
    ----------
    tmp_dict: Dict[str, Any]
        Dictionary that is extended
    value: Any
        Possible values for categorical hyperparameters, or range of values for
        int/float hyperparameters, respectively.

    Returns
    -------
    Dict[str, Any]
        A dictionary that can be added to a list of hyperparameters.
    """

    if isinstance(value, list) and isinstance(value[0], bool):
        tmp_dict["type"] = "categorical"
        tmp_dict["choices"] = [True, False]
    elif isinstance(value, list) and isinstance(value[0], str):
        tmp_dict["type"] = "categorical"
        tmp_dict["choices"] = value
    elif isinstance(value, list) and len(value) != 2:
        raise ValueError(
            "Unsupported parameter input: the float/int parameters "
            + "in the HP_space_ranges file must give a list with two "
            + f"entries that function as a range, but got {value}"
        )
    elif isinstance(value, list) and isinstance(value[0], float):
        tmp_dict["type"] = "uniform_float"
        tmp_dict["log"] = False
        tmp_dict["lower"] = value[0]
        tmp_dict["upper"] = value[1]
    elif isinstance(value, list) and isinstance(value[0], int):
        tmp_dict["type"] = "uniform_int"
        tmp_dict["log"] = False
        tmp_dict["lower"] = value[0]
        tmp_dict["upper"] = value[1]
    else:
        raise ValueError(
            f"Unsupported parameter input: make sure that the parameters "
            + "in the HP_space_ranges file always give a list of booleans or "
            + "strings, or a range of integers or floats (technically also"
            + " a list)"
        )

    # making an empty default entry for the current hyperparameter
    tmp_dict["default"] = None

    return tmp_dict


def create_hp_space(ranges_path: Path, defaults_path: Path) -> Path:
    """Saves a hyperparameter space that is compatible with the JSON reader of the ConfigSpace
    API. The file has JSON format and is being saved to the config/hyperparameter_optimization
    directory.

    The file is created from two files: a hp_space file that defines the ranges of the
    hyperparameters and an options file that defines the default values of the
    hyperparameters. Both are in JSON format.

    Parameters
    ----------
    ranges_path : Path
        Path to the ranges for the hyperparameters.
    defaults_path : Path
        Path to the default values for the hyperparameters.
    """

    # initialize the resulting dictionary that will be saved as a JSON file
    result_dict: Dict[str, List[Any]] = {
        "hyperparameters": [],
        "conditions": [],
        "forbiddens": [],
    }

    # add the hyperparameters as a list of dictionaries to the respective entry in the
    # result_dict
    result_dict["hyperparameters"] = _add_hyperparameters(ranges_path, defaults_path)

    # add conditions as a list of dictionaries to the respective entry in the
    # result_dict
    result_dict["conditions"] = _add_conditions(result_dict["hyperparameters"])

    # save the generated hp_space file
    save_data(result_dict, ranges_path.parents[0] / "hp_space.json")

    logger.info("hp_space generated")

    return ranges_path.parents[0] / "hp_space.json"


# CURRENTLY ONLY FOR DEBUGGING
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    description_r = "json file that defines the ranges of the hyperparameters"
    description_d = "json file that defines the default values of the hyperparameters"
    parser.add_argument("-r", "--ranges_file", help=description_r, type=str)
    parser.add_argument("-d", "--defaults_file", help=description_d, type=str)
    args = parser.parse_args()
    return args


# CURRENTLY ONLY FOR DEBUGGING
if __name__ == "__main__":
    args = parse_arguments()
    ranges = get_absolute_path(f"config/hyperparameter_optimization/{args.ranges_file}")
    defaults = get_absolute_path(f"config/options/{args.defaults_file}")
    create_hp_space(ranges, defaults)
