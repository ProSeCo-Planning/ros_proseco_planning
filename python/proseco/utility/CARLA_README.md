 
# Start (headless) Carla Server

## Install the Carla Module

**See:** [Carla-Simulator Docs](https://carla.readthedocs.io/en/latest/start_quickstart/#carla-installation) for Installation guide

## Setup a Python 3.7 env

Carla requires Python==3.7 and is thus incompatible with the rest of this ProSeCo package.

```bash
# Setup venv
python3.7 -m venv carla_env
source carla_env/bin/activate

# Install required python packages
pip install pygame numpy
# Install carla from egg
python -m easy_install /opt/carla-simulator/PythonAPI/carla/dist/carla-x.x.x-py3.7-linux-x86_64.egg
```

## Start Server

```bash
export SDL_VIDEODRIVER=dummy
export DISPLAY=
CarlaUE4.sh -opengl
```

<u>Flags:</u>

**-carla-rpc-port:** TCP port to listen to (default: 2000)

**-quality-level:** quality level of the simulator (Low, Epic)

## Reference:

**See:** [Carla Homepage](http://carla.org/)


# Run CARLA visualization

```bash
python carla_visualization.py
```


## Flags:

-> default values at the bottom of the script

**-config:** Path to scenario .json-file

**-host:** IP of the host server

**-port:** TCP port to listen to

**-scene:** Location where the scenario is running (see location at top of script)

**-resolution:** Camera window resolution

**-fov:** Field of camera view

**-location:** Position of the camera relative to an agent

**-rotation:** Rotation of the camera relative so its center

**-record_path:** Directory where the video is saved (ATTENTION: all files named "image_*.png" will be deleted!)

**--recording:** enables video generation

## Possible Errors
* `pygame.error: No available video device`
    * occurs when running the carla simulator without a display. Using a dummy video driver should resolve this issue: `export SDL_VIDEODRIVER=dummy`
* `RuntimeError: Spawn failed because of collision at spawn position`
    * occurs when the previous simulation hasn't been terminated properly. Restart the carla server.

# CARLA Dimensions

## Agents: Tesla Model 3

Length: 4.90

Width:  2.06

Height: 1.48

## Obstacles: VW T2

Length: 4.52

Width:  2.08

Height: 2.02

## Lane

Width: 3.5m

# Handling/Remarks

* Scenes are defined in a list at the top of the script by (Map, x, y, reverse)
    * trajectories must always be oriented along the x- or y-axis (values must be increasing)
    * reverse: trajectories are oriented along y-axis
* Agents have distinctive colors (max. 8), obstacles colors are random
    * colors of agents are defined at the top of the script
* Creating scenarios with MCTS:
    * Stick to the Dimensions mentioned above
    * Simulate scenarios with at least 20 fps (timesteps = 0.05s)