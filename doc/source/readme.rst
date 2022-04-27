ROS Interface for ProSeCo Planning - Probabilistic Semantic Cooperative Planning
================================================================================

Table of Contents
-----------------

-  `ProSeCo Planning - Probabilistic Semantic Cooperative
   Planning <#proseco---probabilistic-semantic-cooperative-planning>`__

   -  `Overview <#overview>`__
   -  `Installation Instructions <#installation-instructions>`__

      -  `Installation (C++) <#installation-(C++)>`__
      -  `Installation Python <#installation-(Python)>`__

   -  `Run Instructions <#run-instructions>`__

      -  `Evaluate <#evaluate>`__
      -  `Analyze <#analyze>`__
      -  `Optimize <#optimize>`__

   -  `Configuration Instructions <#configuration-instructions>`__

      -  `Options <#options>`__
      -  `Scenarios <#scenarios>`__
      -  `Evaluator <#evaluator>`__

   -  `Formatting <#formatting>`__
   -  `Documentation <#documentation>`__
   -  `Profiling <#profiling>`__

ProSeCo Planning library
------------------------

The ProSeCo Planning C++ library can be found
`here <https://git.scc.kit.edu/atks/dfg/proseco_planning>`__.

Overview
--------

::

   ├── CMakeLists.txt                               # Project level CMake file
   ├── README.md                                    # This file
   ├── doc                                          # Documentation folder
   ├── config                                   # Configurations folder
   │   ├── evaluator                            # For the evaluator
   │   ├── hyperparameter_optimization          # For the hyperparameter optimization
   │   ├── options                              # For the different algorithmic options
   │   └── scenarios                            # For the different scenarios
   ├── python
   |    └── proseco                              # ProSeCo Planning Python package
   |        ├── evaluator                        # Module for the evaluation of the algorithm
   |        ├── dashboard                        # Module for visualization of the evaluator results
   |        ├── hyperparameter_optimization      # Module for the optimization of the algorithm's hyperparameters
   |        ├── inverse_reinforcement_learning   # Module for the learning of cost functions
   |        ├── testing                          # CI testing script for benchmarking the performance of changes
   |        ├── tests                            # Unit tests for the evaluator and the dashboard
   |        ├── utility                          # Visualization scripts and import helpers
   |        └── visualization                    # Visualization scripts for scenario runs
   ├── include
   │   └── ros_proseco_planning                     # Header files
   └── src                                          # Source files

Installation Instructions
-------------------------

ProSeCo Planning utilizes ROS `Noetic
Ninjemys <http://wiki.ros.org/Distributions>`__ on Ubuntu 20.04. You can
install the library and the corresponding ROS package following the
instructions below.

Installation (C++)
~~~~~~~~~~~~~~~~~~

Without the ProSeCo workspace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Source ROS ``. /opt/ros/${ROS_DISTRO}/setup.bash``

2. Create a *catkin workspace* folder:

   .. code:: bash

      cd ~ && mkdir -p proseco_ws/src

3. Clone
   `ros_proseco_planning <https://git.scc.kit.edu/atks/dfg/ros_proseco_planning>`__
   and
   `proseco_planning <https://git.scc.kit.edu/atks/dfg/proseco_planning>`__
   into your *catkin workspace* folder.

4. Build the catkin workspace with

   .. code:: bash

      cd proseco_ws
      catkin_make_isolated --cmake-args -DCMAKE_BUILD_TYPE=RELEASE

5. Source the package file ``. proseco_ws/devel_isolated/setup.bash``

6. Follow the `usage <#usage>`__ instructions to verify the
   installation.

With the ProSeCo workspace
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to use the Visual Studio Code workspace, it is recommended
to follow the instructions here:
https://git.scc.kit.edu/atks/dfg/proseco_ws

Installation (Python)
~~~~~~~~~~~~~~~~~~~~~

Running ``python3 -m pip install`` in ros_proseco_planning/python
installs the `ProSeCo Python
package <https://git.scc.kit.edu/atks/dfg/ros_proseco_planning/-/blob/develop/python/proseco>`__.

Run Instructions
----------------

If you haven’t already done so, source the environment with

.. code:: bash

   . proseco_ws/devel_isolated/setup.bash

an make sure the Python package is installed and the environment in
which it is installed is activated.

This is **always necessary** if you create a new console and want to use
ProSeCo Planning, also make sure a ``roscore`` is currently running.

Evaluate
~~~~~~~~

.. code:: bash

   cd python/proseco && python evaluator/evaluator.py -c config.json -y -s

Analyze
~~~~~~~

.. code:: bash

   cd python/proseco && python dashboard/index.py

Optimize
~~~~~~~~

.. code:: bash

   cd python/proseco && python hyperparameter_optimization/optimize.py -f optimizer -c config.json

Configuration Instructions
--------------------------

The behavior of the algorithm can be completely configured by changing
the CUE/JSON based configuration files.

Options
~~~~~~~

The options determine the algorithm’s configuration.

Scenarios
~~~~~~~~~

The scenarios describe different scenarios the algorithm can be
evaluated on.

Evaluator
~~~~~~~~~

The evaluation configuration determine the evaluation that is being
performed.

Formatting
----------

C++
~~~

All ``.cpp`` and ``.h`` files must be formatted using the
``.clang-format`` style file. A suitable command on the top level
directory is:
``find . -regex '.*\.\(cpp\|h\)' -exec clang-format -style=file -i {} \;``

Python
~~~~~~

All ``.py`` files should be formatted using
`black <https://github.com/psf/black>`__. Once it’s installed, you can
invoke it on files or directories using

.. code:: bash

   black target

In case you don’t want the automatic formatter to change things for
readability purposes, you can mark these code blocks with ``# fmt: off``
and ``# fmt: on``.

**Example:**

Declaration of a 2D numpy array with matrix-like indentation

.. code:: python

   # fmt: off
   x = np.array([[1, 2, 3],
                 [4, 5, 6]], np.int32)
   # fmt: on

*Note that both ``# fmt`` commands have to be on the same level of
indentation*

Black and Visual Studio Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the variable ``"python.formatting.provider": "black"`` in your
``settings.json``. Now you can automatically format your file with
``CTRL + SHIFT + I``.

JSON and HTML
~~~~~~~~~~~~~

All ``.json`` and ``.html`` files must be formatted using
`Prettier <https://prettier.io>`__.

CUE
~~~

All ``.cue`` files must be formatted using ``cue fmt <file_name>``.

Documentation
-------------

C++ Documentation
~~~~~~~~~~~~~~~~~

| The documentation can be generated using doxygen.
| ``cd doc && doxygen Doxyfile``

Python Documentation
~~~~~~~~~~~~~~~~~~~~

ProSeCo Planning uses `sphinx <https://www.sphinx-doc.org/en/master/>`__
to automatically generate html documentation files for the ProSeCo
Planning Python package. Everything to build the docs is set up in the
``doc`` directory. ProSeCo Planning uses the Scipy sphinx theme which
can be found on
`GitHub <https://github.com/scipy/scipy-sphinx-theme>`__.

| The documentation can be generated using sphinx.js
| ``cd doc && python generate_docs.py``

Sphinx should now start building the documentation and show information
in your shell. Once it’s done you can access the built files in the
``doc/build/html/`` folder.

You can use ``make clean`` from the ``doc`` directory to clean the build
directory.

**Note**: If you want to fully purge all doc files you also have to
delete all generated ``.rst`` files in the ``doc/source`` directory
except ``index.rst`` and ``readme.rst``.

Profiling
---------

The resulting binary can be profiled using:

1. ``valgrind --tool=callgrind --callgrind-out-file=callgrind.out --instr-atstart=no ./proseco_planning_node ros_proseco_planning_node options_basic.json sc01.json``
2. ``kcachegrind callgrind.out``
