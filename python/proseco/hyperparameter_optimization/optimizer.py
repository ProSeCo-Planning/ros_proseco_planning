import sys
import math
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

from proseco.hyperparameter_optimization.createHPSpace import create_hp_space
from proseco.hyperparameter_optimization.analyze_optimizer_results import analyze

# Evaluator
from proseco.evaluator.evaluator import evaluate, Head
from proseco.evaluator.util import flatten_dictionary, nest_dictionary
from proseco.utility.io import (
    get_absolute_path,
    add_extension,
    get_user_temp_dir,
    load_data,
    save_data,
    create_timestamp,
)
from proseco.utility.ui import get_logger

# SMAC
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario
from ConfigSpace.configuration_space import Configuration
from ConfigSpace.read_and_write import json as configspace_json
from smac.initial_design.default_configuration_design import DefaultConfiguration

# Visualizations
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({"figure.autolayout": True})
import seaborn as sns
import logging
import oapackage
import wandb


class Optimizer:
    def __init__(self, args, debug=False):
        self.logger = get_logger("ProSeCo Optimizer")
        if debug:
            self.logger.setLevel(logging.DEBUG)
        self.name = args["name"]
        self.time_string = create_timestamp()
        self.set_paths()
        # Optimizer program arguments
        self.args = args["optimizer"]
        # Name of the base config file for the evaluator
        self.config_name = self.args["base_config"]
        # Open the configuration files
        self.config, self.options, self.HP_space_options = self.open_config()
        self.HP_space, self.mapping_dict = self._init_hp_space()

        # Initial random configuration
        x_ini = dict(self.HP_space.sample_configuration().get_dictionary())
        # Initial arguments instance for head creation
        evaluator_args = self.create_args(x_ini)
        # Head instantiation
        self.head = Head(args=evaluator_args, time_string=self.time_string)

        self.iteration_count = 0
        self.incumbent = None

    def create_output_directory(self) -> Path:
        """Creates the output directory for the optimizer. All output files will be saved here.

        Returns
        -------
        Path
            The output directory for the optimizer.
        """
        output_directory = self.dir_name / "output" / f"{self.time_string}_{self.name}"
        output_directory.mkdir(parents=True)

        return output_directory

    def create_smac_output_directory(self) -> Path:
        """Creates the output directory for smac.

        Returns
        -------
        str
            The output directory for smac.
        """
        smac_output_directory = self.output_directory / "smac"
        smac_output_directory.mkdir(parents=True)

        return smac_output_directory

    def create_graphs_output_directory(self) -> Path:
        """Creates the output directory for the graphs.

        Returns
        -------
        Path
            The output directory for the graphs.
        """
        graphs_output_directory = self.output_directory / "graphs"
        graphs_output_directory.mkdir(parents=True)

        return graphs_output_directory

    def set_paths(self) -> None:
        """Sets frequently used paths."""
        self.dir_name = Path(__file__).parent.resolve()
        self.logger.debug(f"Setting directory path to {self.dir_name}")
        self.config_path = get_absolute_path("config")
        self.logger.debug(f"Setting config path to {self.config_path}")
        self.options_config_path = get_absolute_path("config/options")
        self.logger.debug(f"Setting options path to {self.options_config_path}")
        self.hpopt_config_path = get_absolute_path("config/hyperparameter_optimization")
        self.logger.debug(f"Setting hpopt config path to {self.hpopt_config_path}")
        self.output_directory = self.create_output_directory()
        self.logger.debug(f"Setting output directory to {self.output_directory}")
        self.smac_output_directory = self.create_smac_output_directory()
        self.logger.debug(
            f"Setting smac output directory to {self.smac_output_directory}"
        )
        self.graphs_output_directory = self.create_graphs_output_directory()
        self.logger.debug(
            f"Setting graphs output directory to {self.graphs_output_directory}"
        )

    def get_best_run_index(self, file_path: Path) -> int:
        # Load optimizer output file and get the index of the best run
        optimizer_result = load_data(file_path)
        # Increment by 1: Evaluator output folders start at 1, not at 0
        # @todo Why not start evaluator output folders at 0?
        return optimizer_result["optimum_index"] + 1

    def _init_hp_space(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Initializes the ConfigSpace object for smac. Builds the hyperparameter space from a file that defines the ranges of the hyperparameters and an options file that defines the default values for the hyperparameters.

        Returns
        -------
        Tuple[Dict[str, Any], Dict[str, Any]]
            [description]
        """
        hp_ranges_path = get_absolute_path(
            f"config/hyperparameter_optimization/{self.args['hyperparameter_space']}"
        )
        hp_defaults_path = get_absolute_path(
            f"config/options/{self.config['options'][0] + '.json'}"
        )
        hp_result_space_path = create_hp_space(hp_ranges_path, hp_defaults_path)

        configspace_dict = load_data(hp_result_space_path)
        with open(hp_result_space_path, "r") as file:
            hp_space = configspace_json.read(file.read())
            mapping_dict = {
                hp["path"]: hp["name"] for hp in configspace_dict["hyperparameters"]
            }

        return hp_space, mapping_dict

    def _init_wandb(self):
        """Initializes weighs and biases."""
        wandb.init(
            project="hpopt",
            entity="proseco",
            config={"args": self.args, "hp_space": self.HP_space_options},
        )
        wandb.run.name = f"{self.time_string}_{self.name}"
        wandb.save(glob_str=str(self.hpopt_config_path / "config.json"), policy="now")
        wandb.save(
            glob_str=str(self.hpopt_config_path / self.args["hyperparameter_space"]),
            policy="now",
        )

        wandb.save(
            glob_str=str(
                self.options_config_path / f"incumbent_options_{self.name}.json"
            ),
            policy="live",
        )

    def get_value_if_key_exists(self, string):
        """Returns the value of a key if the key exists.

        Parameters
        ----------
        string: str
            key

        Returns
        -------
        string:
            Value if the key exists
        """

        if string in self.mapping_dict.keys():
            return self.mapping_dict[string]
        else:
            return None

    def open_config(self):
        """Opens the evaluator configuration, the options, and the hp_space .json files

        Returns
        -------
        Dict[str, Any], Dict[str, Any], Dict[str, Any]
            Dictionaries of the config json, options json and HP Space json files.
        """

        evaluator_config = load_data(self.config_path / "evaluator" / self.config_name)

        options = load_data(
            self.options_config_path
            / add_extension(evaluator_config["options"][0], "json")
        )

        hyperparameter_space = load_data(
            self.hpopt_config_path / self.args["hyperparameter_space"]
        )

        return evaluator_config, options, hyperparameter_space

    def analyze_results(self, results, simple=True):
        """Analyzes the results returned by the Head node.

        Opens the nested list data structure (further described in the README)
        and derives a nested dictionary from it (also described in the README).

        Parameters
        ----------
        results: list
            Multidimensional list returned by the evaluator.
        simple  : bool
            Controls whether higher order moments are to bo calculated and added
            to the dictionary.

        Returns
        -------
        Dict[str, Any]
            if simple == False: Results dictionary.
        float, list
            if simple == True: Mean value over all evaluated options-scenario
            instances, list containing all the individual values of the
            options-scenario instances.
        """
        optimization_function = self.args["optimization_function"]
        weights = self.args["multicriterial_weights"]

        if optimization_function == "all":
            optimization_function = [
                "carsCollided",
                "carsInvalid",
                "desiresFulfilled",
                "finalstage",
                "maxSimTimeReached",
                "normalizedCoopRewardSum",
                "normalizedEgoRewardSum",
            ]
        elif isinstance(optimization_function, str):
            optimization_function = [optimization_function]

        hyperparameter_names = self.HP_space.get_hyperparameter_names()
        result_dict = {}
        # Build the result_dict structure: {option_i_uuid: {scen_i_uuid: {<data>}, <data_i>}
        arr = np.array(results, dtype=object)
        self.logger.info(f"Shape: {np.shape(arr)}")

        if simple:
            iterator = results
        else:
            iterator = tqdm(results, desc="Analyzing results")

        for result in iterator:

            uuid = None

            for element in result:
                if element[0] == "uuid.json":
                    uuid = element
                    break

            options_uuid = uuid[1]["options_uuid"]
            tuple_uuid = uuid[1]["options_scenario_uuid"]

            if options_uuid not in list(result_dict.keys()):
                result_dict.update({options_uuid: {tuple_uuid: {"raw_data": {}}}})
            if tuple_uuid not in list(result_dict[options_uuid].keys()):
                result_dict[options_uuid].update({tuple_uuid: {"raw_data": {}}})

            # For every list inside (or str for ip address) inside one result list
            for file in result:
                if isinstance(file, str):
                    break
                elif isinstance(file, list):
                    filename = file[0]
                    if filename == "result.json":
                        factor = 1.0

                        # Check that all optimization_function values are valid
                        for k in optimization_function:
                            assert file[1][k] is not None

                        # If the raw data dictionary is empty
                        if result_dict[options_uuid][tuple_uuid]["raw_data"] == {}:
                            new_dict = {
                                k: [file[1][k] / factor] for k in optimization_function
                            }

                        # If the raw data dictionary already has entries
                        else:
                            new_dict = {
                                k: v / factor
                                for k, v in file[1].items()
                                if k in optimization_function
                            }
                            new_dict = {
                                k: v + [new_dict[k]]
                                for k, v in result_dict[options_uuid][tuple_uuid][
                                    "raw_data"
                                ].items()
                            }

                        result_dict[options_uuid][tuple_uuid]["raw_data"].update(
                            new_dict
                        )

                    elif filename == "scenario_output.json" and "scenario" not in list(
                        result_dict[options_uuid][tuple_uuid].keys()
                    ):
                        scenario = file[1]["name"]
                        result_dict[options_uuid][tuple_uuid].update(
                            {"scenario": scenario}
                        )

                    elif filename == "options_output.json" and "options" not in list(
                        result_dict[options_uuid].keys()
                    ):
                        options = flatten_dictionary(file[1])
                        options_values = [
                            v
                            for k, v in options.items()
                            if self.get_value_if_key_exists(string=k)
                            in hyperparameter_names
                        ]
                        result_dict[options_uuid].update({"options": options_values})

        # Iterate over result_dict in order to calculate the mean and moments
        result_dict = analyze(dic=result_dict, simple=simple)

        # Get the mean of the option of the selected optimization_function
        result = []
        for k in result_dict.keys():
            result_k = np.array(
                [result_dict[k]["mean_data"][func] for func in optimization_function]
            )
            result_k = np.dot(result_k, np.array(weights))
            result.append(result_k)

        self.logger.debug(result_dict)

        if simple:
            # Return a scalar function value of the first options instance, and
            # a 1-d list of all mcts values which where used to calculate the
            # scalar mean of result
            return (
                np.mean(result),
                result_dict[list(result_dict.keys())[0]]["raw_data"][
                    optimization_function[0]
                ],
            )
        else:
            # Return the entire result dictionary, see the README for a detailed overview.
            return result_dict

    def optimize(self):
        """Optimizes the objective function on the HP Space.

        Performs a random search or a smac optimization cycle according to the
        program parameters. If smac is selected, a directory starting with
        smac3-... is created in the current directory and contains all relevant
        optimizer cycle information. More information can be found following
        the smac documentation.

        Returns
        -------
        2d-list, 1d-list, 1d-list, 2d-list
            X 2d-list containing the instantiations of each HP. One column
            corresponds to one HP, one row to one configuration.
            y 1d-list, i.e. the vector of objective function values for each
            optimizer iteration.
            t 1d-list, i.e time in second of each iteration.
            y_detailed 2d-list, each row corresponds to the evaluations of one
            iteration. The mean value of the i-th row corresponds to the i-th
            entry in the y vector.
        """

        optimizer = self.args["type"]
        seed = self.args["seed"]
        number_iterations = self.args["number_iterations"]
        x_ini = list(self.HP_space.sample_configuration().get_dictionary().values())

        # Initialize wandb here to prevent unnecessary wandb init in analyzer
        self._init_wandb()

        scenario = Scenario(
            {
                "run_obj": "quality",  # we optimize quality (alternatively runtime)
                "runcount-limit": number_iterations,  # max. number of function evaluations
                "cs": self.HP_space,  # configuration space
                "deterministic": "true",
                "limit_resources": False,
                "rand_prob": 0.2,
                "output_dir": str(self.smac_output_directory),
            }
        )

        y_detailed = []
        y = []
        X = []
        t = []

        y_same_options = []

        def validate_base_incumbent(X, max_index):
            """Validates the obtained results.

            At the end of one optimizer cycle, the baseline and the best
            configuration can be reevaluated multiple times in order to reduce
            the objective function uncertainty.

            Parameters
            ----------
            X: 2d-list
                Configuration 2d-list, where each row corresponds to one
                configuration.
            max_index: int
                Index of the configuration which performed best. Is equivalent
                to the index of the iteration which yielded the highest objective
                function value.

            Returns
            -------
            2d-list
                List containing the results of the validation procedure. The
                first row corresponds to the objective function value of the
                baseline configuration, the second to the objective function
                values of the incumbent configuration.
            """
            x_baseline = X[0]
            x_incumbent = X[max_index]

            dict_baseline = dict(zip(self.mapping_dict.values(), x_baseline))
            dict_incumbent = dict(zip(self.mapping_dict.values(), x_incumbent))

            config_baseline = Configuration(
                configuration_space=self.HP_space, values=dict_baseline
            )
            config_incumbent = Configuration(
                configuration_space=self.HP_space, values=dict_incumbent
            )
            configurations = [config_baseline, config_incumbent]

            y_ensemble = []
            y_d_ensemble = []

            iterations_multiplier = self.args["validate_baseline_incumbent"][
                "iterations_multiplier"
            ]

            # Perform the validating evaluations.
            for i, configuration in enumerate(configurations):
                for j in range(iterations_multiplier):
                    y_iter = f(configuration)
                    y.append(y_iter)
                    y_ensemble.append(y_iter)

            # List [[results of baseline (1-d List for every result)], [results of incumbent (1-d list for every result)]]
            y_d_base_inc_list = y_detailed[self.args["number_iterations"] :]
            y_d_base_inc_list = list(
                np.reshape(
                    y_d_base_inc_list, (2, iterations_multiplier * len(y_detailed[0]))
                )
            )

            return y_d_base_inc_list

        def same_options():
            """Evaluates the same configuration multiple times.

            Performs multiple objective function evaluations of the same
            configuration.
            """

            for i in range(10):
                y_same_options.append([])
                x_rand = self.HP_space.sample_configuration()
                for j in range(number_iterations):
                    time_start = datetime.now()
                    x_rand = self.HP_space.get_default_configuration()
                    y_iter = f(x_rand)
                    y.append(y_iter)
                    X.append(x_rand)
                    t.append((datetime.now() - time_start).total_seconds())
                    y_same_options[i].append(y_iter)
                y_detailed.clear()
                y.clear()
                X.clear()
                t.clear()

        def random_search():

            """Iteratively samples random configurations until the budget is exhausted"""
            """Iteratively samples random configurations.

            Iteratively samples random configurations from the HP Space until
            the specified number of iterations has been processed.
            """

            def sample(i):
                """Samples configuration.

                Sample the default configuration if first iteration. Else,
                sample randomly.

                Parameters
                ----------
                i: int
                    Iterations index.

                Returns
                -------
                configSpace.Configuration
                    Sampled Configuration from the HP Space.
                """
                if i == 0:
                    return self.HP_space.get_default_configuration()
                else:
                    return self.HP_space.sample_configuration()

            for i in range(number_iterations):
                time_start = datetime.now()
                x_rand = sample(i)
                y_iter = f(x_rand)
                y.append(y_iter)
                X.append(list(x_rand.get_dictionary().values()))
                t.append((datetime.now() - time_start).total_seconds())

        def smac_4BB():
            """Optimizes using SMAC_4BB optimizer.

            Creates a directory starting with smac3-... and outdumps the
            optimization data in it. For more information on how smac works,
            see the documentation.
            """

            smac = SMAC4BB(
                scenario=scenario,
                rng=np.random.RandomState(seed),
                tae_runner=f,
                initial_design=DefaultConfiguration,
            )

            smac.optimize()
            smac_get_results()

        def smac_4HPO():
            """Optimizes using SMAC_4HPO optimizer.

            Creates a directory starting with smac3-... and outdumps the
            optimization data in it. For more information on how smac works,
            see the documentation.
            """

            smac = SMAC4HPO(
                scenario=scenario,
                rng=np.random.RandomState(seed),
                tae_runner=f,
                initial_design=DefaultConfiguration,
            )

            smac.optimize()
            smac_get_results()

        def smac_get_results():
            """Fetches the results from the created smac directory.

            Fetches the json files from the latest smac directory and converts
            it into the X (configuration 2d-list), y (results list), and t the
            time 1d-list.
            """

            assert (
                len(list(self.smac_output_directory.iterdir())) == 1
            ), "Found more than one smac run directory."

            trajectory_path = (
                list(self.smac_output_directory.iterdir())[0] / "runhistory.json"
            )

            data = load_data(trajectory_path)
            iterator = data["data"]
            for index, iteration in enumerate(data["data"]):
                y.append(iteration[1][0])
                x = list(data["configs"][str(index + 1)].values())
                X.append(x)
                t.append(iteration[1][1])

        def f(x):
            """Objective function


            Parameters:
            ----------
                x: {} dictionary of all values from a HP configuration
            Output:
                result: float function value
            """
            """Objective function.   

            Calculates the objective function value of one MCTS run. 
            
            Parameters  
            ----------  
            x: configSpace.Configuration
                Configuration object to be evaluated.

            Returns  
            -------  
            float
                Sample mean of the objective function values over all scenarios
                and repetitions.
            """

            def save_incumbent_in_wandb():
                """Saves the incumbent options in wandb and in the smac_output
                directory

                Parameters:
                -----------
                    result: the result of the objective function for the
                    incumbent run
                """
                path = (
                    get_user_temp_dir()
                    / "proseco_evaluator_output"
                    / f"{self.time_string}_{self.config['evaluation_name']}"
                    / f"iteration_{self.iteration_count}"
                )
                self.logger.debug(
                    f"Fetching the incumbent options from {list(path.iterdir())[0]}"
                )

                incumbent_options = load_data(
                    list(path.iterdir())[0] / "options_output.json"
                )

                save_data(
                    incumbent_options,
                    self.options_config_path / f"incumbent_options_{self.name}.json",
                )

            dictionary = dict(x.get_dictionary())
            args = self.create_args(dictionary)
            results = evaluate(head=self.head, args=args, outside_eval=True)
            result, result_list = self.analyze_results(results)
            # Transforming absolute objective function value to
            # value relative to theoretical optimum
            result_list = [self.args["best_function_value"] - x for x in result_list]
            result = self.args["best_function_value"] - result

            erg = 0
            y_detailed.append(list(result_list))
            for d in y_detailed:
                for dd in d:
                    erg += dd
            erg = erg / 60
            self.logger.info(f"Evaluation returned value: {result}.")
            # @todo check type of result
            if (self.iteration_count == 0) or (result < self.incumbent):
                self.incumbent = result
                save_incumbent_in_wandb()

            wandb.log(
                {"objective_function": result, "incumbent": self.incumbent},
                step=self.iteration_count,
            )
            self.iteration_count += 1

            return result

        # Which optimizer is to be run.
        if "random" == optimizer:
            run_optimizer = random_search
        elif "smac_4BB" == optimizer:
            run_optimizer = smac_4BB
        elif "smac_4HPO" == optimizer:
            run_optimizer = smac_4HPO
        elif "same_options" == optimizer:
            run_optimizer = same_options
        else:
            self.logger.error(
                "Optimizer type in the program.json was missspecified: "
                + optimizer
                + "\Must be in ['random', 'smac_4BB', 'smac_4HPO', 'same_options']"
            )
            sys.exit(1)

        # Run the optimizer
        run_optimizer()

        try:
            assert len(y) == len(t)
        except AssertionError as error:
            self.logger.error(error)

        # Perform secondary operations if the corresponding option is set.
        if self.args["baseline_incumbent_plot"]:
            index = y.index(min(y))
            if self.args["validate_baseline_incumbent"]["active"]:
                y_list = validate_base_incumbent(X, index)
            else:
                y_list = [y_detailed[0], y_detailed[index]]
            self.plot_baseline_incumbent(y_list[0], y_list[1], optimizer)
            if self.args["validate_baseline_incumbent"]["active"]:
                last_index = (
                    2
                    * self.args["validate_baseline_incumbent"]["iterations_multiplier"]
                )

                y = y[:-last_index]
                y_detailed = y_detailed[:-last_index]

        assert len(y) == len(y_detailed)

        if self.args["efficient_frontier_plot"]:
            var_list = [
                np.var(y_detailed[i], ddof=1) ** (0.5) for i in range(len(y_detailed))
            ]
            self.plot_efficient_frontier(y, var_list, optimizer)

        self.write_output(X, y, y_detailed, t, optimizer, index)

        assert len(y) == len(y_detailed)
        assert len(t) == len(y)

        return list(X), list(y), list(t), list(y_detailed)

    def plot_baseline_incumbent(self, list_a, list_b, name_alg):
        """Plots overlapping histograms and boxplots.

        Plots one histogram and one boxplot for each objective function sample
        within one single figure and saves it in the graphs directory.

        Parameters
        ----------
        list_a: 1d-list
            1d-list of the objetive function samples of the baseline
            configuration.
        list_b: 1d-list
            1d-list of the objective function smaples of the incumbent
            configurations.
        name_alg: str
            Name of the optimization algorithms from which both samples originate.
        """
        data = pd.DataFrame(
            list(zip(list_a, list_b)), columns=["baseline", "incumbent"]
        )
        # https://stackoverflow.com/questions/57458789/get-bin-width-used-for-seaborn-plot
        bin_width = (
            2
            * (np.quantile(list_a, 0.75) - np.quantile(list_a, 0.25))
            / np.cbrt(len(list_a))
        )
        right_stop = max(list_a) + bin_width
        left_stop = min(list_a) - bin_width
        # right_stop = 1751
        bin_list = np.arange(left_stop, right_stop, 25)

        # Check if bin_list is empty -> Plot only box plot
        if len(bin_list) < 2:
            figure, ax = plt.subplots(1)
            sns.boxplot(data=data, ax=ax, orient="h")
        else:
            # bin_list not empty -> Plot box plot and histogram
            figure, (ax_box, ax_hist) = plt.subplots(
                2, sharex=True, gridspec_kw={"height_ratios": (0.2, 0.80)}
            )
            ax_hist.set(
                xlabel="Objective function",
                ylabel="Frequency",
            )
            ax_hist.set_xlim(left=left_stop, right=right_stop)
            figure.suptitle(
                "Baseline vs. incumbent: "
                + name_alg
                + ", n_b = "
                + str(len(list_a))
                + ", n_i = "
                + str(len(list_b))
            )
            sns.boxplot(data=data, ax=ax_box, orient="h")
            kde = True
            if math.isclose(np.var(list_a), 0) or math.isclose(np.var(list_b), 0):
                kde = False
            sns.histplot(data=data, ax=ax_hist, bins=bin_list, kde=kde)

        figure.savefig(
            self.graphs_output_directory / f"baseline_vs_incumbent_{name_alg}.png"
        )

    def plot_efficient_frontier(self, mu_list, var_list, name):
        """Plots a scatter plot.

        PLots a scatter plot, where the y-axis is the objective function sample
        mean and x-axis the objective function sample standard deviation. Points
        are red and connected if they are pareto optimal, blue if not. Saves
        the figure in the graphs directory.

        Parameters
        ----------
        mu_list: 1d-list
            List of the objective function sample means.
        var_list: 1d-list
            List of the objective function sample standard deviations.
        name: str
            Optimizer name which generated the objective function samples.
        """

        mu_list = [-x for x in mu_list]
        var_list = [-x for x in var_list]

        datapoints = np.array([mu_list, var_list], dtype="object")
        pareto = oapackage.ParetoDoubleLong()

        for ii in range(0, datapoints.shape[1]):
            w = oapackage.doubleVector((datapoints[0, ii], datapoints[1, ii]))
            pareto.addvalue(w, ii)
        lst = pareto.allindices()
        optimal_datapoints = datapoints[:, lst]

        list_a = list(optimal_datapoints[0])
        s_list_a = list(sorted(list_a))
        list_b = list(optimal_datapoints[1])
        optimal_datapoints = [
            list(sorted(list_a)),
            [list_b[list_a.index(s_list_a[i])] for i in range(len(list_a))],
        ]

        mu_list = [-x for x in mu_list]
        var_list = [-x for x in var_list]

        for i in range(len(optimal_datapoints)):
            for ii in range(len(optimal_datapoints[i])):
                optimal_datapoints[i][ii] = -optimal_datapoints[i][ii]

        figure, ax = plt.subplots()
        figure.suptitle("Pareto optimal frontier, n = " + str(len(var_list)))
        ax.plot(var_list, mu_list, ".b", label="Not pareto optimal")
        ax.plot(
            optimal_datapoints[1],
            optimal_datapoints[0],
            ".r-",
            label="Pareto optimal frontier",
        )
        ax.set(xlabel="Estimated std. deviation", ylabel="Objective function")
        ax.legend()
        figure.savefig(self.graphs_output_directory / f"efficient_frontier_{name}.png")

    def write_output(self, X, y, y_d, t, optimizer_name, optimum_index):
        """Dumps the output of one optimizer cycle in a json file.

        Creates a dictionary contianing X, y, t, y_d (configuration 2d-list,
        results 1d-list, time 1d-list, and 2d-results list). Refer to README
        for more precise description.

        Parameters
        ----------
        X: 2d-list
            Configuration 2d-list.
        y: 1d-list
            Results 1d-list.
        t: 1d-list
            Evaluation times 1d-list.
        y_d: 2d-list
            2d-results list.
        optimizer_name: str
            Name of the optimizer.
        optimum_index: int
            Index of the iteration where the best objective function value has
            been obtained.
        """
        ttest = ttest_ind(y_d[0], y_d[optimum_index], equal_var=False)
        output_dictionary = {
            "name": optimizer_name,
            "X": X,
            "y": y,
            "y_d": y_d,
            "t": t,
            "optimum_index": optimum_index,
            "welch_test_stat_pvalue": [ttest.statistic, ttest.pvalue],
            "stddev_baseline": np.var(y_d[0]) ** 0.5,
            "stddev_incumbent": np.var([optimum_index]) ** 0.5,
        }
        file_name = self.output_directory / f"{optimizer_name}_{len(y)}_evals.json"
        save_data(output_dictionary, file_name)

    def create_args(self, x):
        """Creates an args object for the evaluator.

        Creates an object from the configuration dictionary x.

        Parameters
        ----------
        x: Dict[str, Any]
            Dictionary with shortened HP names as keys and instantiations as its
            values.

        Returns
        -------
        Args
            Args object containing the attributes necessary for the evaluator.
        """

        # create a dict with the keys of mapping_dict and the values of x for all
        # entries in x
        new_x = {}
        for path, hp_name in self.mapping_dict.items():
            if hp_name in x.keys():
                new_x[path] = [x[hp_name]]

        nested_dict = nest_dictionary(new_x)

        for key, value in self.config["options_alterations"]["compute_options"].items():
            if key not in nested_dict["compute_options"]:
                nested_dict["compute_options"].update({key: value})
        self.config["options_alterations"].update(nested_dict)

        self.logger.info(nested_dict)
        return EvaluatorArgs(self.config)


@dataclass
class EvaluatorArgs:
    config: Dict
    yes: bool = False
    output: Optional[bool] = None
    no_dashboard: bool = False
    debug: bool = False
    no_summary: bool = True
    address: Optional[str] = None
