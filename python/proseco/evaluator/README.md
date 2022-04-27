# ProSeCo Evaluator

The ProSeCo evaluator is a script used to compute multiple runs of the ProSeCo planner in parallel on a ray cluster.

## Setup
Ensure that the the proseco python package is installed
```bash
cd /ros_proseco_planning/python && pip install -e .
```
### Running on a Cluster (optional)
1. Source the ray.bash file in your .bashrc (`/ros_proseco_planning/python/proseco/evaluator/ray.bash`).  
*Note: You will likely need to adjust the paths in the [ray.bash](ray.bash) file.*  

2. Create your cluster.
*Note: You will likely need to adjust [autoscaler.yaml](autoscaler.yaml) for your setup.*
```bash
ray up autoscaler.yaml -y
```
3. Ensure that all your workers have spawned by navigating to the dashboard [(http://localhost:8265/)](http://localhost:8265/).  
*Note: The port is specified in the autoscaler.yaml.*
## Start
To start the evaluator, ensure that a `roscore` is running on the local machine (the cluster spawns the roscore automatically).
```bash
python evaluator.py -c <config.json>
```

Visit the [Ray-Documentation](https://docs.ray.io/en/master/) for more information.

### Configuration file

In order to specify the scenarios and options to be evaluated, the ProSeCo evaluator uses a config file ([/ros_proseco_planning/config/evaluator/config.json](../../../config/evaluator/config.json)). Every value inside of the .json file, which is susceptible to be altered within the evaluation, has to be arranged inside a list, even in the case of a single element. Examples for this are the chosen the number of iterations, the discount factor and any other option that can be specified in the options file. Below is an example:

```json
{
    "evaluation_name": "example",
    "options": ["example_options"],
    "scenarios": [ "sc01", "sc02", "sc03", "sc04", "sc05", "sc06", "sc07", "sc08", "sc09", "sc10"],
    "number_runs": 1,
    "options_alterations": {
        "compute_options": {
            "discount_factor": [0.7, 0.8, 0.9],
            "n_iterations": [500, 1000, 2000],
            "random_seed": [ 331, 650, 28, 134, 198, 882, 167, 572, 230, 163, 483, 185, 8, 968, 306, 653, 493, 684, 272, 470, 359, 460, 857, 933, 101],
        }
    },
    "scenario_alterations": {},
    "ray_cluster": {
        "address": null,
        "max_workers": 0
    }
}
```

### Evaluate() function

This function allows to start the evaluation of a configuration dictionary (or .json file), given the already running ray cluster and a roscore on each node. It returns results in a list according to the following structure:

```python
[[<self.ip>_i, [<filename>_j, <json-dict or csv-list>], ..., ['uuid.json', {<uuid>}], ... ]
```

Each result (ProSeCo planner run) is indexed with i, each file inside the result as j.

### Error occurrences

It is possible to analyze crashes resulting from the ros_proseco_planning C++ code. First, make sure that your machine has `apport` enabled (using Ubuntu) and start the evaluator using the -d flag (enables debug mode).   
The ray worker will then call `ulimit -c unlimited` for the process it is running in, which in turn enables the creation of core dumps. Whenever the C++ code crashes, a new file `core` should be created either in the `proseco_ws` folder or the folder where the `proseco_planning_node` is located.   
Now run `gdb ./devel_isolated/ros_proseco_planning/lib/ros_proseco_planning/proseco_planning_node core` (assuming core is inside `proseco_ws`) and type `bt`, which should give you the full stacktrace. You can quit gdb using CTRL+D.

## Result structure

The results are saved at the following place using the following structure:

```bash
/tmp/$USER/proseco_planning_output/<timestamp_start_evaluator.py>/<result>_<i-th_ProSeCo_run>_on_<last_two_digits_of_worker_ip_address>_inum<iteration-number>
```

In this folder, you will find all the output files (`.json` & `.msgpack`) of the i-th ProSeCo run, as well as a universally unique identifier, which correspond to a specific scenario and options tuple. This is useful when trying to find all the evaluations of one particular tuple. The iteration number designates the amount of times different config files were evaluated during the lifetime of one evaluator head instance.
