# ROS Interface for ProSeCo Planning - Probabilistic Semantic Cooperative Planning

## Table of Contents

- [ProSeCo Planning - Probabilistic Semantic Cooperative Planning](#proseco---probabilistic-semantic-cooperative-planning)
  - [Overview](#overview)
  - [Setup](#setup)
  - [Run Instructions](#run-instructions)
    - [Evaluate](#evaluate)
    - [Analyze](#analyze)
    - [Optimize](#optimize)
  - [Configuration Instructions](#configuration-instructions)
    - [Options](#options)
    - [Scenarios](#scenarios)
    - [Evaluator](#evaluator)
  - [Formatting](#formatting)
  - [Documentation](#documentation)
  - [Profiling](#profiling)

## ProSeCo Planning C++ library

The ProSeCo Planning C++ library can be found [here](https://github.com/ProSeCo-Planning/proseco_planning).

## Overview

    ├── README.md                                # This file
    ├── CMakeLists.txt                           # Project level CMake file
    ├── doc                                      # Documentation
    ├── config                                   # Configuration files
    │   ├── evaluator                            # Configuration of the evaluator
    │   ├── hyperparameter_optimization          # Configuration of the hyperparameter optimization
    │   ├── options                              # Configuration of the planning algorithm
    │   └── scenarios                            # Configuration of the different scenarios
    ├── python
    |    └── proseco                              # ProSeCo Planning Python package
    |        ├── evaluator                        # Module for the evaluation of the algorithm
    |        ├── dashboard                        # Module for the visualization of the evaluator results
    |        ├── hyperparameter_optimization      # Module for the optimization of the hyperparameters
    |        ├── testing                          # End-to-end tests for the ProSeCo Planning C++ library
    |        ├── tests                            # Unit tests for the ProSeCo Planning Python package
    |        ├── utility                          # Module with utility functions
    |        └── visualization                    # Module with visualization functions
    ├── include
    │   └── ros_proseco_planning                  # Header files
    └── src                                       # Source files

## Setup
Please follow the instructions in the [ProSeCo Planning workspace](https://github.com/ProSeCo-Planning/proseco_workspace#setup) to get started with the library.

## Run Instructions

1. Source the environment: `. proseco_ws/devel_isolated/setup.bash`
1. Activate the Python virtual environment
1. Start a ROS core: `roscore`

### Evaluate

```bash
cd python/proseco && python evaluator/evaluator.py -c config.json -y -s
```

### Analyze

```bash
cd python/proseco && python dashboard/index.py
```

### Optimize

```bash
cd python/proseco && python hyperparameter_optimization/optimize.py -f optimizer -c config.json
```

## Configuration Instructions

The behavior of the algorithm can be completely configured by changing the CUE/JSON based configuration files.

### Options

The options determine the algorithm's configuration.

### Scenarios

The scenarios describe different scenarios the algorithm can be evaluated on.

### Evaluator

The evaluation configuration determines the evaluation that is being performed.

## Formatting
The ProSeCo Planning workspace provides a [script](https://github.com/ProSeCo-Planning/proseco_workspace/blob/main/format_all.bash) to format all `.cpp`, `.h` and `.py` files.

### C++

All `.cpp` and `.h` files must be formatted using [clang-format](https://clang.llvm.org/docs/ClangFormat.html).

### Python

All `.py` files must be formatted using [black](https://github.com/psf/black).

### JSON and HTML

All `.json` and `.html` files must be formatted using [Prettier](https://prettier.io).

### CUE

All `.cue` files must be formatted using [cue](https://cuelang.org/), e.g. `cue fmt <file_name>`.

## Documentation

### C++ Documentation

The documentation can be generated using doxygen.
`cd doc && doxygen Doxyfile`

### Python Documentation

The documentation can be generated using sphinx.js  
`cd doc && python generate_docs.py`

## Profiling

The resulting binary can be profiled using:

1. `valgrind --tool=callgrind --callgrind-out-file=callgrind.out --instr-atstart=no ./proseco_planning_node ros_proseco_planning_node example_options.json sc01.json`
2. `kcachegrind callgrind.out`
