"""The io module contains functions for path, directory and file interaction."""

from typing import Any, Dict, List, Union
import json
import msgpack
import rospkg
import getpass
from pathlib import Path
import datetime


def get_number_digits(number: int) -> int:
    """Gets the number of digits in a number.

    Parameters
    ----------
    number : int
        The number.

    Returns
    -------
    int
        The number of digits.
    """
    return len(str(number))


def create_timestamp() -> str:
    """Creates a timestamp using the current time.

    Returns
    -------
    str
        The timestamp.
    """

    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_list_of_directories(
    directory_path: Union[str, Path], descending=False
) -> List[Path]:
    """Gets a list of directories in a directory.

    Parameters
    ----------
    directory_path : Union[str, Path]
        The path to the directory.

    Returns
    -------
    List[Path]
        The list of directories.
    """
    if isinstance(directory_path, str):
        directory_path = Path(directory_path)

    return sorted(
        [directory for directory in directory_path.iterdir() if directory.is_dir()],
        reverse=descending,
    )


def get_list_of_files(directory_path: Union[str, Path], descending=False) -> List[Path]:
    """Gets a list of files in a directory.

    Parameters
    ----------
    directory_path : Union[str, Path]
        The path to the directory.

    Returns
    -------
    List[Path]
        The list of files.
    """
    if isinstance(directory_path, str):
        directory_path = Path(directory_path)

    return sorted(
        [file for file in directory_path.iterdir() if file.is_file()],
        reverse=descending,
    )


def get_ros_pack_path() -> Path:
    """Gets the absolute path to the ROS proseco_planning package.

    Returns
    -------
    Path
        The path to the ROS proseco_planning package.
    """
    return Path(rospkg.RosPack().get_path("ros_proseco_planning"))


def get_absolute_path(file_path: Union[str, Path]) -> Path:
    """Gets an absolute path for a file path relative to the ROS package.

    Parameters
    ----------
    file_path : Union[str, Path]
        The relative path of the file to the ROS package.

    Returns
    -------
    Path
        The absolute path.
    """

    if isinstance(file_path, str):
        file_path = Path(file_path)

    ros_pkg_path = get_ros_pack_path()
    return ros_pkg_path / file_path


def get_user_temp_dir() -> Path:
    """Gets the path to the users temporary directory, located at /tmp/${USER}.

    Returns
    -------
    Path
        The path to the temporary directory of the user.
    """
    return Path(f"/tmp/{getpass.getuser()}")


def add_extension(file_path: Union[str, Path], extension: str) -> Union[str, Path]:
    """Adds the extension to the file path if it is not present.

    Parameters
    ----------
    file_path : Union[str, Path]
        The path to the file.
    extension : str
        The extension to add.

    Returns
    -------
    Union[str, Path]
        The path with the extension.
    """

    if isinstance(extension, str):
        if not extension.startswith("."):
            extension = "." + extension
    else:
        raise TypeError(f"Invalid extension type: {type(extension)}")

    if isinstance(file_path, Path):
        if not file_path.suffix == extension:
            file_path = file_path.with_suffix(extension)
    elif isinstance(file_path, str):
        if not file_path.endswith(extension):
            file_path += extension
    else:
        raise TypeError(f"Invalid file path type: {type(file_path)}")

    return file_path


def load_json(file_path: Union[str, Path]) -> Union[Dict[str, Any], List[Any]]:
    """Loads a json file and returns the data.

    Parameters
    ----------
    file_path : Union[str, Path]
        The path to the json file.

    Returns
    -------
    Dict
        The data in the json file.
    """
    with open(file_path, "r") as file:
        return json.load(file)


def load_msgpack(file_path: Union[str, Path]) -> Union[Dict[str, Any], List[Any]]:
    """Loads a msgpack file and returns the data.

    Parameters
    ----------
    file_path : Union[str, Path]
        The path to the msgpack file.

    Returns
    -------
    Dict
        The data in the msgpack file.
    """
    with open(file_path, "rb") as file:
        return msgpack.load(file)


def load_data(file_path: Union[str, Path]) -> Union[Dict[str, Any], List[Any]]:
    """Loads the data from a file depending on the file extension i.e. json or msgpack.

    Parameters
    ----------
    file_path : Union[str, Path]
        The path to the file.

    Returns
    -------
    Union[Dict[str, Any], List[Any]]
        The data in the file.
    """
    if isinstance(file_path, dict):
        raise TypeError(f"Invalid file path type: {type(file_path)}")
        return file_path

    # Convert to string, e.g. if Path or PosixPath is passed
    file_path = str(file_path)

    if file_path.endswith(".msgpack"):
        return load_msgpack(file_path)
    elif file_path.endswith(".json"):
        return load_json(file_path)
    else:
        raise ValueError(f"Invalid file extension: {file_path}")


def save_msgpack(data: Any, file_path: Union[str, Path]) -> None:
    """Saves a msgpack file.

    Parameters
    ----------
    data : Any
        The data to save.
    file_path : Union[str, Path]
        The path to the msgpack file.
    """
    with open(file_path, "wb") as file:
        msgpack.dump(data, file)


def save_json(data: Any, file_path: Union[str, Path], sort=True) -> None:
    """Saves a json file.

    Parameters
    ----------
    data : Any
        The data to save.
    file_path : Union[str, Path]
        The path to the json file.
    sort : bool
        Whether to sort the json data by its keys.
    """
    with open(file_path, "w") as file:
        json.dump(data, file, indent=2, sort_keys=sort)


def save_data(
    data: Union[Dict[str, Any], List[Any]], file_path: Union[Path, str], sort=True
) -> None:
    """Saves the data to a file depending on the file extension i.e. json or msgpack.

    Parameters
    ----------
    data : Union[Dict[str, Any], List[Any]]
        The data to save.
    file_path : str
        The path to the file.
    sort : bool
        Whether to sort the data (if possible, e.g. json by its keys), defaults to True.
    """
    # Convert to string, e.g. if Path or PosixPath is passed
    file_path = str(file_path)

    if file_path.endswith(".msgpack"):
        save_msgpack(data, file_path)
    elif file_path.endswith(".json"):
        save_json(data, file_path, sort)
    else:
        raise ValueError(f"Invalid file extension: {file_path}")


def load_scenario(scenario_name: str) -> Dict[str, Any]:
    """Loads the scenario from the scenario directory.
    Parameters
    ----------
    scenario_name : str
        The name of the scenario to load.
    Returns
    -------
    Dict[str, Any]
        The scenario.
    """
    scenario_path = get_absolute_path(
        f"config/scenarios/{add_extension(scenario_name, 'json')}"
    )
    return load_data(scenario_path)


def load_options(options_name: str) -> Dict[str, Any]:
    """Loads the options from the options directory.
    Parameters
    ----------
    options_name : str
        The name of the options to load.
    Returns
    -------
    Dict[str, Any]
        The options.
    """
    options_path = get_absolute_path(
        f"config/options/{add_extension(options_name, 'json')}"
    )
    return load_data(options_path)
