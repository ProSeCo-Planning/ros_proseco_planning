"""The ui module contains functions for interacting with the user."""

from typing import Union
from pathlib import Path
import http.server
import socketserver
import socket
import logging

from proseco.utility.io import get_list_of_directories


def get_logger(
    logger_name: str, log_level: int = logging.INFO, create_handler=True
) -> logging.Logger:
    """Creates a logger with the given name and logging level.

    Parameters
    ----------
    logger_name : str
        The name of the logger.
    log_level : int, optional
        The logging level, by default logging.INFO
    create_handler : bool, optional
        Whether to create a handler for the logger (this is needed if it is the first instance of the logger), by default False.
    Returns
    -------
    logging.Logger
        The logger.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    if create_handler:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    return logger


def get_log_level(logger_name: str) -> int:
    """Gets the logging level of a logger.

    Parameters
    ----------
    logger_name : str
        The name of the logger.

    Returns
    -------
    int
        The logging level.
    """
    return logging.getLogger(logger_name).getEffectiveLevel()


def let_user_select_subdirectory(directory_path: Union[str, Path]) -> Path:
    """Let the user select a directory from a list of directories.

    Parameters
    ----------
    directory_path : Union[str, Path]
        The path to the file.

    Returns
    -------
    Path
        The path to the directory.
    """
    if isinstance(directory_path, str):
        directory_path = Path(directory_path).resolve()
    directory_paths = get_list_of_directories(directory_path, descending=True)

    try:
        if len(directory_paths) == 0:
            raise FileNotFoundError(f"No directories found in {directory_path}")
    except FileNotFoundError:
        print(
            f"No directories found in {directory_path}, please select from parent folder {directory_path.parent.resolve()}"
        )
        return let_user_select_subdirectory(directory_path.parent.resolve())

    for i, directory_path in enumerate(directory_paths):
        print(f"{i}: {directory_path.stem}")

    while True:
        try:
            user_input = int(input("Select a directory: "))
            if user_input < 0 or user_input >= len(directory_paths):
                raise ValueError(f"Invalid directory: {user_input}")
            return directory_paths[user_input]
        except ValueError:
            print("Invalid directory. Please try again.")


def start_web_server(directory_path: Union[str, Path], port: int = 9999) -> None:
    """Starts a web server in a specific directory.

    Parameters
    ----------
    directory : Union[str, Path]
        The directory to serve files from.
    port : int
        The port to use for the web server.
    """
    if isinstance(directory_path, str):
        directory_path = Path(directory_path)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory_path), **kwargs)

    httpd = socketserver.TCPServer(("", port), Handler)
    print(
        f"Serving files from {directory_path.relative_to(Path().resolve())} at http://{socket.gethostname()}:{port}"
    )
    httpd.serve_forever()
