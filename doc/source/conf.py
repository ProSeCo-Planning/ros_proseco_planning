# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from proseco.utility.io import get_absolute_path

current_dir = os.path.dirname(__file__)
target_dir = str(get_absolute_path("python"))
sys.path.insert(0, target_dir)


# -- Project information -----------------------------------------------------

project = "ProSeCo"
copyright = "2020, Karl Kurzer"
author = "Karl Kurzer"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.napoleon", "recommonmark"]

# Generate Docs also from the markdown files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

## Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [target_dir + "/setup.py"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_logo = os.path.join(current_dir, "../images/logo.png")
html_theme = "nature"
html_theme_path = ["."]


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []
