# ProSeCo Python Package for the C++ ProSeCo Planning Library
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

The ProSeCo Python package provides additional functionality that is complementing the C++ library, that focuses on real-time capabilities.
1. The evaluator `/evaluator` enables detailed evaluation of the results of the ProSeCo Planning algorithm.
1. The dashboard `/dashboard` can be used for rich visualization
1. Automatic hyperparameter optimization can be achieved with `/hyperparameter_optimization`.
1. The folder `/utility` contains helper scripts providing usefull functions used by the other scripts.  

Below is an overview of the available scripts and a quick explanation on how to run the scripts.

## Installation of dependencies
The hyperparameter optimization and inverse reinforcement learning code have non-Python dependencies.

- The package `smac` requires `SWIG` version 3.x.
- The package `pycairo` is dependent on `libcairo2`.

You can install all dependencies with:
```bash
sudo apt install swig3.0 libcairo2-dev libjpeg-dev libgif-dev
```

## Installation of necessary Python Packages

It's best to use a virtual environment for the Python package. To avoid conflicts you should let the installer set up
the necessary dependencies instead of installing them beforehand. For this follow the steps below.

1. Create a virtual environment:
   ```bash
   python3 -m venv proseco
   ```

1. Activte the virtual environment:
   ```bash
   . proseco/bin/activate
   ```

1. Upgrade pip:
   ```bash
   pip install --upgrade pip
   ```

1. Install wheel and numpy first (required by some dependencies of the ProSeCo package):
   ```bash
   pip install wheel numpy~=1.19.2
   ```

1. Install the ProSeCo Python package. Assuming you're in this folder:
   ```bash
   cd ros_proseco_planning/python && pip install -e .
   ```

## How to use the python scripts

To run the scripts simply run the command as shown in the example below form the sourced virtual environment.

```bash
python evaluator.py -c <config.json>
```

The image below illustrates how the different scripts work together and which configuration files they need.

![Scripts relationships](visualize_python_scripts.svg?s=100)

## Available scripts

### `evaluator`

- [evaluator.py](./evaluator/evaluator.py)
    - runs the ProSeCo Planning algorithm for specifief scenarios with varied parameters
    - stores all results under `/tmp/$USER/proseco_evaluator_output`
    - can be subclassed and called by other python scripts, such as the hyperparameter optimization

### `hyperparameter_optimization`

Optimize given parameters of the MCTS.

- [optimization.py](./hyperparameter_optimization/optimization.py)
    - optimizes given parameters

### `dashboard`

Webservice for visualizing the evaluation results using plotly and dash.

- [index.py](./dashboard/index.py)
    - main page

### `visualization`

Visualizations using matplotlib based on scenario and trajectory files.

- scenario_visualizer.py
    - visualizes a scenario and optionally a driven trajectory
- scenario_video_visualizer.py
    - generates a 2d video from a given trajectory file
- trajectory_velocity_visualizer.py
    - visualizes the trajectory together with the velocities of each agent
- trajectory_reward_visualizer.py
    - visualizes the trajectory together with the rewards of each agent
    
### `utility`

Various scripts containing helper functions which are used in the other scripts.
