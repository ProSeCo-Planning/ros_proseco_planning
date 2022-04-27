"""This script generates the documentation for the ProSeCo python package. It calls the sphinx-apidoc command and then executes the make command."""

import os
from pathlib import Path

from proseco.utility.ui import start_web_server


def doc_generator():
    # generate .rst files from the source code of the ProSeCo package
    # -f overwrites existing files, -o sets the target directory
    # "/source" is the target directory, "../python/proseco/" is the path to the ProSeCo package
    os.system(
        "sphinx-apidoc -f -o source ../python/proseco ../python/proseco/utility/*carla_visualization*  && make html"
    )


if __name__ == "__main__":
    doc_generator()
    html_directory = Path(__file__).parent.resolve() / "build/html"
    # start_web_server(html_directory)
