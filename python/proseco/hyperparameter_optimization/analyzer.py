import os
from pathlib import Path
import json
import re
from proseco.utility.ui import let_user_select_subdirectory
from tqdm import tqdm
from itertools import chain, combinations
from collections import OrderedDict
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from proseco.utility.io import (
    load_data,
    get_user_temp_dir,
    add_extension,
)

# Evaluator
from proseco.hyperparameter_optimization.optimizer import Optimizer

# Visualizations
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({"figure.autolayout": True})
import seaborn as sns


def calculate_significance_dict(comparison_list_a, comparison_list_b):
    """Generates the significance dictionary.

    Takes two samples as its input and creates a dictionary depicting

    Returns
    -------
    Dict[str, Any]
        Opened program json file.
    """
    comparison_list_a = list(comparison_list_a)
    comparison_list_b = list(comparison_list_b)
    mean_a = np.mean(comparison_list_a)
    mean_b = np.mean(comparison_list_b)
    ttest = ttest_ind(comparison_list_a, comparison_list_b, equal_var=False)
    test_less = mannwhitneyu(comparison_list_a, comparison_list_b, alternative="less")
    test_greater = mannwhitneyu(
        comparison_list_a, comparison_list_b, alternative="greater"
    )
    significance_dict = {
        "mean_a": mean_a,
        "sample_size_a": len(comparison_list_a),
        "mean_b": mean_b,
        "sample_size_b": len(comparison_list_b),
        "ttest_welch_stat_pvalue": [ttest.statistic, ttest.pvalue],
        "mwu_test_h1_a_smaller_b_stat_pvalue": [
            test_less.statistic,
            test_less.pvalue,
        ],
        "mwu_test_h1_a_bigger_b_stat_pvalue": [
            test_greater.statistic,
            test_greater.pvalue,
        ],
    }

    return significance_dict


class Analyzer:
    def __init__(self, args, optimizer: Optimizer):
        """Instantiates an Analyzer object.

        Initializes the attributes of the Args object.

        Parameters
        ----------
        optimizer: Optimizer
            Optimizer object.
        """

        self.args = args["analyzer"]
        self.optimizer = optimizer
        self.time_string = optimizer.time_string
        self.dir_name = os.path.dirname(__file__)

    def open_evaluation(self):
        """Creates a result dictionary from nested evaluator directories.

        Creates a result dictionary from the nested directories contained in the
        evaluator results. The resulting dictionary is saved in the analyzer
        directory.

        Returns
        -------
        Dict[str, Any]
            Results dictionary.
        """

        evaluation_name = self.args["evaluation_name"]
        evaluation_dir = get_user_temp_dir() / "proseco_evaluator_output"
        splitnames = evaluation_name.split("/")
        output_path = let_user_select_subdirectory(
            Path(__file__).parent.resolve() / "output"
        )

        if len(splitnames) == 1:
            # Case 1: Evaluation datestring is given, e.g. "2021_10_26__16_01_18"
            eval_datestring = splitnames[0]
            # Check that it is correct and construct path to best run
            if re.match(r"\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}", eval_datestring):
                # Path to the optimizer output directory
                opt_output_path = (
                    Path(self.pack_path)
                    / "python"
                    / "proseco"
                    / "hyperparameter_optimization"
                    / "optimizer_output"
                )

                # Get the output file given the datestring
                try:
                    # Would raise index error if file doesn't exist
                    opt_output_file = list(
                        opt_output_path.glob(f"{eval_datestring}*.json")
                    )[0]
                except IndexError:
                    # Raise a more expressive error when the file doesn't exist
                    raise FileNotFoundError(
                        "No optimizer output file could be found for the given datestring."
                    )

                # Load optimizer output file and get the index of the best run
                optimizer_result = load_data(opt_output_file)
                # Increment by 1: Evaluator output folders start at 1, not at 0
                best_run_idx = optimizer_result["optimum_index"] + 1

                # Get the evaluator output folder for the given datestring
                try:
                    eval_folder = list(evaluation_dir.glob(f"*_{eval_datestring}*"))[0]
                except IndexError:
                    raise FileNotFoundError(
                        "No evaluator output files were found for the given datestring."
                    )

                # Fetch the directory of the run with the best index
                run = Path(eval_folder) / f"result_inum_{best_run_idx}"
            else:
                raise ValueError("Given evaluation datestring has the wrong format.")
        elif len(splitnames) == 2:
            # Case 2: Full evaluation directory is given.
            # e.g. "2021_10_26__16_01_18_example_hpopt_evaluation/result_inum_1"
            # Construct the directory and pass
            run = evaluation_dir / evaluation_name
        else:
            raise ValueError(
                "Given evaluation name has an incorrect format. Please supply either an evaluation datestring or an evaluation folder"
            )

        dir_list = os.listdir(run)

        result_list = []

        # for result_i directory in all results
        for directory in tqdm(dir_list, desc=f"Loading results from {evaluation_name}"):

            sub_path = os.path.join(run, directory)
            uuid_path = os.path.join(sub_path, "uuid.json")
            result_sublist = []

            # Iterate over every file
            files = [
                f
                for f in os.listdir(sub_path)
                if os.path.isfile(os.path.join(sub_path, f))
            ]

            for file_name in files:

                file_path = os.path.join(sub_path, file_name)
                # Only consider result. scenario_output. and options_output.json
                if file_name == "uuid.json" or file_name.endswith(".msgpack"):
                    continue
                data = load_data(file_path)
                if file_name == "scenario_output.json":
                    name = data["name"]
                    data.clear()
                    data["name"] = name
                result_sublist.append([file_name, data])
            data = load_data(uuid_path)
            result_sublist.append(["uuid.json", data])

            result_list.append(result_sublist)
        result_dictionary = self.optimizer.analyze_results(result_list, simple=False)

        # Saving result dictionary
        path = "analyzer"
        sub_path = os.path.join(path, self.time_string + ".json")

        # Dumps the analyzed data dictionary
        os.makedirs(path, exist_ok=True)
        with open(sub_path, "w") as j:
            json.dump(result_dictionary, j, indent=4)

        return result_dictionary

    def plot_result_dict(self, result_dict):
        """Plots one result dictionary.

        Creates a histrogram + boxplot for each options-scenario tuple as well
        as for each instance. In addition, one figure where all options-scenario
        tuples are overlapping is created.

        Parameters
        ----------
        result_dict: Dict[str, Any]
            Results dictionary.
        """

        def plot_raw_data(raw_dic, name):
            """Plots the results dictionary.

            Parameters
            ----------
            raw_dict: Dict[str, Any]
                Dictionary with all the raw results of one options or options-scenario tuple instance.
            name: str
                Name of the plot.
            """
            weights = self.optimizer.args["multicriterial_weights"]
            optimization_functions = self.optimizer.args["optimization_function"]
            for index, optimization_function in enumerate(optimization_functions):
                raw_dic[optimization_function] = [
                    x * weights[index] for x in raw_dic[optimization_function]
                ]
            x = [
                self.optimizer.args["best_function_value"]
                - sum([raw_dic[k][i] for k in optimization_functions])
                for i in range(len(raw_dic[optimization_function]))
            ]

            figure, (ax_box, ax_hist) = plt.subplots(
                2, sharex=True, gridspec_kw={"height_ratios": (0.20, 0.80)}
            )
            # Iterate over each of the two
            ax_box.set_xlim(left=0.0, right=int(max(x)) + 25)
            bins_list = np.arange(start=0, stop=int(max(x)) + 25, step=30)
            ax_hist.set(xlabel="Objective function", ylabel="Frequency")
            sns.boxplot(x=x, ax=ax_box)
            plt.hist(x, density=True, bins=bins_list)
            fig_folder = Path(self.dir_name) / "graphs"
            if not fig_folder.exists():
                fig_folder.mkdir(exist_ok=True)
            figure.savefig(fig_folder / f"{self.time_string}_{name}.png")

        # Iterate over every options uuid of result dictionary
        for i, k in enumerate(result_dict.keys()):
            r_d = result_dict[k]["raw_data"]
            m_d = result_dict[k]["mean_data"]
            v_d = result_dict[k]["variance_data"]
            not_relevant_keys = [
                "options",
                "mean_data",
                "variance_data",
                "skewness_data",
                "kurtosis_data",
                "jarque_bera_stat_pvalue",
                "scenario",
                "raw_data",
            ]
            # Iterate over relevant tuple_keys (not in not_relevant_keys, that is the uuids)
            tuple_keys = np.setdiff1d(list(result_dict[k].keys()), not_relevant_keys)
            scenario_names = []
            for kk in tuple_keys:
                scenario_names.append(result_dict[k][kk]["scenario"])
                raw_data = result_dict[k][kk]["raw_data"]
                if self.args["tuple_plot"]:
                    plot_raw_data(
                        raw_data,
                        result_dict[k][kk]["scenario"] + "_" + k,
                    )
            raw_data = result_dict[k]["raw_data"]
            scenarios_string = "-".join(scenario_names)
            if self.args["options_plot"]:
                plot_raw_data(
                    result_dict[k]["raw_data"],
                    "option_id_" + k,
                )
            # @TODO Raw data von allen tuples zusammenstellen und plotten lassen

    def plot_trajectory(self, y, t, names):
        """Plots the optimization trajectory.

        Plots the optimization trajectory, in function of time as well as in
        function of the iterations number. The y-axis is the best previously
        found objective function value. Saves the figures under the graphs
        directory.

        Parameters
        ----------
        y: 1d-list
            Results 1d-list.
        t: 1d-list
            Iterations time 1d-list.
        names: list
            1d-list of strings, which describe the compared optimization
            algorithms.
        """

        def plot(figure_tuple, ax_tuple, y, t, name):
            """Actual plot function.

            Plots one optimization cycle.

            Parameters
            ----------
            figure_tuple: tuple
                Tuple containing two matplotlib figure objects.
            ax_tuple: bool
                Tuple containing two axis objects corresponding to the figures.
            y: 1d-list
                1d-list results.
            t: 1d-list
                1d-list of iteration times.
            name: str
                Name of the optimization algorithm having produced the plotted
                optimizer cycle.

            Returns
            -------
            tuple, tuple
                figure_tuple as well as ax_tuple containing the optimization
                cycle trajectory.
            """

            t_aggregated = list([sum(t[:i]) for i in range(1, len(t) + 1)])
            y_decreasing = list([min(y[:i]) for i in range(1, len(y) + 1)])
            iterations_list = list(np.arange(1, len(y) + 1))

            (line_1,) = ax_tuple[0].step(
                iterations_list, y_decreasing, label=name, where="post"
            )
            (line_2,) = ax_tuple[1].step(
                t_aggregated, y_decreasing, label=name, where="post"
            )

            return figure_tuple, ax_tuple

        if not isinstance(y[0], list):
            y_list = [y]
            t_list = [t]
            names = [names]
            iterations = list(np.arange(1, len(y) + 1, 1))
        else:
            y_list = y
            t_list = t
            figure_agg, ax_agg = plt.subplots()
            figure_agg.suptitle("Optimizer comparison")
            iterations_list = list(np.arange(1, len(y[0]) + 1, 1))

        # Powersets, except that the empty set is not included (if len(y_list)==1, tuple_comp = (1,))
        tuple_comp = tuple(OrderedDict.fromkeys((1, len(y_list))))

        power_y = list(
            tuple(chain.from_iterable(combinations(y_list, i))) for i in tuple_comp
        )
        power_t = list(
            tuple(chain.from_iterable(combinations(t_list, i))) for i in tuple_comp
        )
        power_n = list(
            tuple(chain.from_iterable(combinations(names, i))) for i in tuple_comp
        )

        # Iterate over every subset of the powerset having cardinality one and max
        for y_tuple, t_tuple, n_tuple in tuple(zip(power_y, power_t, power_n)):
            figure_1, ax_1 = plt.subplots()
            figure_2, ax_2 = plt.subplots()
            figure_tuple = (figure_1, figure_2)
            ax_tuple = (ax_1, ax_2)
            ax_tuple[0].set(ylabel="Objective function", xlabel="Iterations")
            ax_tuple[1].set(ylabel="Objective function", xlabel="Time [s]")
            ax_tuple[0].set_xscale("log")
            ax_tuple[1].set_xscale("log")
            ax_tuple[0].set_yscale("log")
            ax_tuple[1].set_yscale("log")
            plot_name = " vs. ".join([n_tuple[i] for i in range(len(n_tuple))])

            for y, t, name in zip(y_tuple, t_tuple, n_tuple):
                figure_tuple, ax_tuple = plot(figure_tuple, ax_tuple, y, t, name)
            figure_tuple[0].suptitle(
                "Optimizer trajectory: " + plot_name + " over iterations"
            )
            figure_tuple[1].suptitle(
                "Optimizer trajectory: " + plot_name + " over time"
            )
            ax_tuple[0].legend()
            ax_tuple[1].legend()

            for fig in figure_tuple:
                fig.savefig(
                    self.optimizer.graphs_output_directory
                    / add_extension(fig.texts[0].get_text(), "png")
                )

    def compare_algorithms(self, args, optimizer):
        """Compares different algorithm optimizer cycles.

        Compares the optimizer cycles of various algorithms and generates
        two dictionaries on their basis, i.e a significance and a comparison
        dictionary. The comparison dictionary contains the information of each
        optimization cycle of each optimizer algorithm, and the significance
        dictionary the pairwise sample means, the welch test statistics as well
        as the mann-whitney-u test statistic. Both are dumped as json files in
        the alg_comparison directory.

        Parameters
        ----------
        args: argparse.Args
            Results dictionary.
        optimizer: optimizer
            Optimizer object.

        Returns
        -------
        Dict[str, Any], Dict[str, Any]
            Comparison and significance dictionaries.
        """

        def plot_pairwise_comparison(comparison_dict):
            """Plots a histogram with two boxplots.

            Plots the histogram of the aggregated samples of each optimization
            algorithm, with two overstanding boxplots. Saves the figure under the
            graphs directory.

            Parameters
            ----------
            comparison_dict: Dict[str, Any]
                Comparison dictionary.
            """

            os.makedirs(self.dir_name + "/graphs", exist_ok=True)

            max_values_tuples = []
            max_values_list = []

            for algorithm in comparison_dict.keys():
                max_values_tuples.append(
                    (algorithm, comparison_dict[algorithm]["max_values"])
                )
                max_values_list.append(comparison_dict[algorithm]["max_values"])
            for x, y in combinations(max_values_tuples, 2):
                name_a = x[0]
                name_b = y[0]
                x = x[1]
                y = y[1]
                name = name_a + "_" + name_b
                figure, (ax_box, ax_hist) = plt.subplots(
                    2, sharex=True, gridspec_kw={"height_ratios": (0.2, 0.80)}
                )
                data = pd.DataFrame(list(zip(x, y)), columns=[name_a, name_b])
                ax_hist.set(xlabel="Objective function", ylabel="Frequency")
                figure.suptitle("Histogram: " + name_a + " vs. " + name_b)
                sns.boxplot(data=data, ax=ax_box, orient="h")
                sns.histplot(data=data, ax=ax_hist, kde=True)

                figure.savefig(
                    self.dir_name
                    + "/graphs/"
                    + self.time_string
                    + "_comparison_pdf_"
                    + name
                    + ".png"
                )
            name = " vs. ".join(list(comparison_dict.keys()))
            figure, ax = plt.subplots()
            figure.suptitle("Box plot: " + name)
            ax.set(xlabel="Objective function")
            sns.boxplot(data=max_values_list, orient="h", ax=ax)
            ax.set_yticklabels(list(comparison_dict.keys()))
            figure.savefig(
                self.dir_name
                + "/graphs/"
                + self.time_string
                + "_comparison_boxplot_"
                + name
                + ".png"
            )

        algorithms = args["optimizer_comparison"]["algorithms"]
        iterations = args["optimizer_comparison"]["iterations_budget"]
        repetitions = args["optimizer_comparison"]["repetitions"]

        # Lists for trajectory plotting: []
        y_traj = []
        t_traj = []
        names = algorithms

        comparison_dict = {}
        y_alg = []
        y_alg_extrema = []
        # Iterate over each algorithm type
        for i, algorithm in enumerate(algorithms):
            t_traj.append([])
            y_traj.append([])
            y_alg.append([])
            y_alg_extrema.append([])
            comparison_dict.update({algorithm: {}})
            optimizer.args.update({"type": algorithm})
            optimizer.args.update({"number_iterations": iterations})
            # Iterate over each repetition
            for j, repetition in enumerate(range(repetitions)):
                # Comment out if you want to debug
                y_traj[i].append([])
                t_traj[i].append([])
                X, y, t, y_d = optimizer.optimize()
                y_traj[i][j].extend(y)
                t_traj[i][j].extend(t)

                # For testing purposes replace the preceding 5 lines with the following four

                # y_d = np.random.normal(300 + i*random.random()*50, 50+i*20, 5000).reshape((10, 500))
                # y = list(np.mean(y_d, axis = 1))
                # y_traj[i].append(y)
                # t_traj[i].append(list(np.arange(len(y))))

                y_d = list(y_d)
                comparison_dict[algorithm].update({str(repetition): y})
                index_y_min = y.index(min(y))
                y_alg[i].extend(y_d[index_y_min])
        # Iterate over each algorithm type
        for i, algorithm in enumerate(algorithms):
            comparison_dict[algorithm].update({"mean": np.mean(y_alg)})
            comparison_dict[algorithm].update({"variance": np.var(y_alg, ddof=1)})
            comparison_dict[algorithm].update({"max_values": y_alg[i]})

        # Calculate significance level
        # All subsets of length 2 of algorithms
        algorithm_comparison_subsets = list(combinations(algorithms, 2))
        significance_dict = {}
        for alg1, alg2 in algorithm_comparison_subsets:
            key = "(" + str(alg1) + "," + str(alg2) + ")"
            comparison_list_a = comparison_dict[alg1]["max_values"]
            comparison_list_b = comparison_dict[alg2]["max_values"]
            sub_significance_dict = calculate_significance_dict(
                comparison_list_a, comparison_list_b
            )
            significance_dict.update({key: sub_significance_dict})

        base_path = os.path.join(
            self.dir_name,
            "alg_comparison",
        )
        os.makedirs(base_path, exist_ok=True)
        path_comp = os.path.join(
            base_path, "alg_comparison_" + self.time_string + ".json"
        )
        path_sign = os.path.join(
            base_path, "alg_significance_" + self.time_string + ".json"
        )
        with open(path_comp, "w") as j:
            json.dump(comparison_dict, j, indent=4)
        with open(path_sign, "w") as j:
            json.dump(significance_dict, j, indent=4)

        # Pairwise comparison of the pdfs of the best found points per algorithm over all repetitions
        if args["optimizer_comparison"]["comparison_plot"]:
            plot_pairwise_comparison(comparison_dict)

        # Iterate over every algorithm output
        for i, (y, t) in enumerate(zip(y_traj, t_traj)):
            # Iterate over every repetition of one algorithm
            for (j, y_grow) in enumerate(y):
                y[j] = [min(y_grow[:k]) for k in range(1, len(y_grow) + 1)]
            y_avg_traj = list(np.average(y, axis=0))
            t_avg_traj = list(np.average(t, axis=0))
            y_traj[i] = y_avg_traj
            t_traj[i] = t_avg_traj

        # Plotting the trajectory
        self.plot_trajectory(list(y_traj), list(t_traj), list(names))

        return comparison_dict, significance_dict
