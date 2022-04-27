# ProSeCo Hyper-Parameter Optimization

The ProSeCo Hyper-Parameter Optimization is a tool suite tailored for tuning the hyperparameter of the ProSeCo planner.

**INSTALLATION**:

- Make sure that the proseco package, Swig, Nodejs, and Phantomjs are installed in order to benefit from the complete tool suite.
- Swig 3.0 is required for proper installation of python packages (`sudo apt install swig3.0`).
- Modify the [config.json](../../../config/hyperparameter_optimization/config.json) file and save it.
- Launch the optimizer from inside the "hyperparameter_optimization" directory: `python optimization.py -c config.json -f <function to be run>`. Possible functions: optimizer, analyzer or optimizer_comparison.

## Before Starting

### Hyperparameter Optimization Configuration

The [config.json](../../../config/hyperparameter_optimization/config.json) file specifies which tool you want to run. At the moment, three different tools are implemented.
It comprises the following information:

```json
{
    "analyzer": {
        "evaluation_name": "2020_09_06__09_52_18_analyzer", # ProSeCo evaluator directory to analyze
        "options_plot": true,
        "tuple_plot": true
        }

    },
    "optimizer": {
        "base_config": "config.json",   # config file used to launch the evaluator (with updated "options_alterations").
        "number_iterations": 1,         # Number of configurations the optimizer will evaluate to find an optimal configuration
        "hyperparameter_space": "example_hp_space.json", # .json file that defines the hyperparameter and bounds
        "baseline_incumbent_plot": true,
        "efficient_frontier_plot": true,
        "optimization_function": [
            "carsCollided",
            "carsInvalid",
            "desiresFulfilled",
            "maxSimTimeReached",
            "normalizedCoopRewardSum",
            "normalizedEgoRewardSum"],  # Metrics to be used in the objective function
        "multicriterial_weights": [-0.2, -0.2, 0.1, -0.1, 0.2, 0.0, 0.1, -0.1], # Weights used to build the scalar product with for the according optimization functions (take the different scales into account!).
        "seed": null                    # Random seed for the optimizer
        "best_function_value": 685,     # Necessary when transforming maximization problem into minimization problem (the the maximum theoretical value reachable by the objective function).
        "type": "smac_4HPO",            # smac_4BB (bayesian optimization with GP), smac_4HPO (Random Forest surrogate function) or random
        "validate_baseline_incumbent": {# Evaluate the the baseline as well as the incumbent configuration multiple times
            "active": false,
            "iterations_multiplier": 10
        }
    },
    "optimizer_comparison": {
        "algorithms": ["smac_4BB", "smac_4HPO", "random"], # Algorithms to be compared
        "comparison_plot": true,
        "iterations_budget": 50,    # Number of steps the optimizer is allowed to use
        "repetitions": 10           # Number of times the optimizer is launched
    }
}
```

#### Optimizer

Launching an optimizer cycle generates a smac output in case a smac-algorithm was chosen.
Furthermore, all options and every tuple is plotted in form of a histogram combined with a boxplot. These plots are located under `graphs/`.

#### Analyzer

Launching an analyzer cycle generates plots in form of a histogram combined with a boxplot for each options and tuple instances. These plots are located under `graphs/`.  
In addition, some key features of the results are saved within a .json file within the `analyzer/` directory, which shares the above detailed results dictionary structure.

#### Optimizer Comparison

Launching an optimizer comparison cycle generates two dictionaries, comprising the following information:
Algorithm comparison:

```python
{
    <algorithm>_k : {
        ...
        "<Iteration number>_i-1": [] # List containing the sample mean of each optimizer iteration (length specified in config.json at "optimizer_comparison/iterations_budget")
        "<Iteration number>_i": [] # i ranging from 1 to the value specified in config.json at "optimizer_comparison/repetitions".
        ...
        "mean": 302.34322       # Sample mean of the single maximum value in every optimizer iteration.
        "variance" : 4032.4235  # Sample variance of the single maximum values in every optimizer iteration.
        "max_values": []        # List containing all the objective function values of the evaluator results (used for plotting purposes only).
    }
    <algorithm>_k+1 : {analogous}
}
```

Algorithm significance:

```python
{
    "(smac_4BB,smac_4HBO)": {           # Pair of compared algorithms
        "mean_a": 303.4849024248869,    # Mean of the maximum values attained in the optimizer in each iteration.
        "sample_size_a": 5000,          # Sample size of all combined maximum values.
        "mean_b": 332.70799471815235,   # Analogous.
        "sample_size_b": 5000,          # Analogous.
        "ttest_welch_stat_pvalue":  # t-Test H0: mu_a = mu_b, first value is test statistic, second p-value.
        [   -23.922155563235147,
            1.1786412697711056e-122
        ],
        "mwu_test_h1_a_smaller_b_stat_pvalue": # Mann-Whitney-U-Test, in case not large enough sample size for central limit theorem. First value is test-statistic, second is p-value.
        [   9194538.0,
            2.3356966466757413e-116
        ],
        "mwu_test_h1_a_bigger_b_stat_pvalue":  # Same information, but regarding the other tail
        [   9194538.0,
            1.0
        ]
    }
}
```

### Hyperparameter Space

The hyperparameter space (hp space) is created internally from two json files: one that defines the hyperparameters as well as their ranges to be optimized in (e.g. [example_hp_space.json](../../../config/hyperparameter_optimization/example_hp_space.json)) and one that defines the default values of the hyperparameters (e.g. [options_basic.json](../../../config/options/options_basic.json)). It is saved as [result_hp_space.json](../../../config/hyperparameter_optimization/result_hp_space.json).  
The default values file are defined in the evaluator configuration (e.g. [options_basic.json](../../../config/options/options_basic.json)). Subgroups of options (see options.cue) are automatically (de)activated if the flag for that group is called "active". If not, the conditions must be hard-coded in [`createHPSpace.py`](createHPSpace.py).

Ranges file structure:

```json
{
  "compute_options": {
    "max_search_depth": [1, 5],
    "discount_factor": [0.1, 0.99],
    "policy_options": {
      "simulation_Policy": ["random", "moderate"],
      "final_selection_policy": [
        "maxActionValue",
        "maxVisitCount",
        "mostTrusted"
      ],
      "policy_enhancements": {
        "progressive_widening": {
          "coefficient": [0.1, 5.0],
          "exponent": [0.1, 0.9],
          "max_depth_pw": [2, 5]
        }
      }
    },
    "uct_cp": [2.0, 8.0]
  }
}
```

**Checklist before starting a hyperparameter optimization**

- [ ] check name, optimization_function, number_iterations, base_config in [config.json](../../../config/hyperparameter_optimization/config.json)
- [ ] check evaluation_name, options, ray_cluster in evaluator [config.json](../../../config/evaluator/config.json))
- [ ] check export, export_format in options
- [ ] check env in
  - [ ] .bashrc
  - [ ] autoscaler.yaml
  - [ ] ray.bash
  - [ ] setup.bash
- [ ] check $ROS_PACKAGE_PATH
- [ ] delete unnecessary evaluations from
  - [ ] /tmp directory
  - [ ] smac_output
  - [ ] W&B
- [ ] start ray cluster
- [ ] run hpopt in tmux

### Result data structure

The results are organized in a dictionary data structure described below. It is generated inside the `optimizer.py -> analyze_results()` method from the list structured output of the ProSeCo evaluator.

```python
{
    <options_uuid>_i-1: { ... },
    <options_uuid>_i: {
        <tuple_uuid>_j-1: { ... }
        <tuple_uuid>_j: {
            raw_data: {
                ...
                <metric>_k-1:  [True, True, False ...], # length n
                <metric>_k:    [334.3429, ...],         # length n
                ...
            },
            mean_data: {
                ...
                <metric>_k-1:  0.66 # calculated over all n runs of one tuple
                <metric>_k:    334.3429,
                ...
            },
            var_data:               { analogous },
            skewness_data:          { analogous },
            kurtosis_data:          { analogous },
            jarque_bera_pvalue:     { analogous },
            scenario:               "sc01"
        },
        mean_data: {
            ...
            <metric>_k-1:   0.3243 # calculated over all tuples of specified options instance
            <metric>_k:     334.3429
            ...
        },
        raw_data: {
            ...
            <metric>_k:     [342.8982, 343.2573, ...] # length n*m
            ...
        },
        var_data:           { analogous },
        skewness_data:      { analogous },
        kurtosis_data:      { analogous },
        jarque_bera_pvalue: { analogous },
        options :           [0.2433, "circleApproximation" ...] # instantiation of alphabetically sorted hyperparameters
    }
}
```

A tuple is a combination of one options instantiation and one scenario instantiation.

- n is the amount of times a tuple was evaluated
- m is the amount of scenarios evaluated
- i is the options index
- j is the scenario (tuple) index

## Analyzing the results

### Weights and Biases

[Weights & Biases](http://www.wandb.ai) is used to track experiments. For each run, a file with the options of the current incumbent is stored. Additionally, it can be found in the `smac_output/` directory. There is also a file named `traj_aclib2.json` stored in the same directory. It provides useful information on when the incumbent changed.

### CAVE

In order to use the CAVE toolsuit to analyze a SMAC optimization cycle, make sure to

1. Have Node.js and PhantomJS (`npm install phantomjs-prebuilt`) installed  
   _Note: Make sure that phantomjs-prebuilt is in your path, e.g. `export PATH=./node_modules/phantomjs-prebuilt/bin:$PATH`_
2. Run the `run_cave_analysis.py` script and select the optimizer run you want to analyze. It will automatically generate a CAVE report and start a web server in the CAVE output directory.