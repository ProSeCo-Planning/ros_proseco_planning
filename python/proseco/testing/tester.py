#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 03.08.2020

@author: Timo Klein
"""

###############################################################################
# This script contains a tester class performing tests for ProSeCo. The main
# tester class is dervied from the Evaluator.
# The evaluation results are evaluated against success rate thresholds specified
# by the config file in this folder.
# The tests are intended to be run in the continuous integration pipeline.
###############################################################################
import argparse
import rospkg
import glob
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional

from proseco.evaluator.head import Head
from proseco.utility.io import load_data, get_ros_pack_path
from proseco.utility.ui import get_logger


class TestFailedError(Exception):
    """Error class to throw when tests failed."""

    def __init__(self, message):
        super().__init__(message)


class ProSeCo_Tester(Head):
    """Main testing class for the ProSeCo Planning library.

    Runs tests using the evaluator from the ProSeCo Python package.
    Test settings are specified in the test_thresholds.json.
    """

    def __init__(
        self,
        *args,
        threshold_file: str,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # change to the base package directory
        os.chdir(get_ros_pack_path())
        self.testPath = Path(os.getcwd()) / "test"

        # set up logging and colored shell output
        self.c_error = "\033[91m \033[1m"
        self.c_warning = "\33[33m \033[1m"
        self.c_okay = " \033[92m \033[1m"
        self.c_reset = " \033[0m"

        self.logger = (
            get_logger("ProSeCo Tester", logging.INFO) if logger is None else logger
        )
        self.thresholds = load_data(threshold_file)

        assert set(self.bulk_config["scenarios"]).issubset(
            set(self.thresholds.keys())
        ), (self.c_warning + "Testing scenarios without threshold set" + self.c)

        self.print_test_settings()

    def run_tests(self) -> None:
        """Run all tests as specified in the ci_config with the thresholds set in test_thresholds.

        Performs the following tasks by calling the respective methods:
            1. Runs the evaluator with the settings specified in
                ros_proseco_planning/config/evaluator/ci_config.json
            2. Evaluates if each run was successful or not.
                carsCollided and carsInvalid constitute failures.
                Everything else is considered a success.
            3. Evaluates the success rates for each scenario agains the thresholds set in
                ros_proseco_planning/test/test_thresholds.json
            4. Prints detailed information about each scenario and failed tests.
            5. Fail the CI if tests are not passed.

        """
        _ = self.start()
        results = self.create_summary(save_to_disk=False)
        # count successes and runs
        scenarios = self.bulk_config["scenarios"]
        no_successes: Dict[str, int] = {scenario: 0 for scenario in scenarios}
        no_runs: Dict[str, int] = {scenario: 0 for scenario in scenarios}

        total_runs = 0
        total_successes = 0
        for run in results:
            # get run information and update result dictionaries
            scen_name, success = self.extract_run_info(run)
            no_successes[scen_name] += success
            total_successes += success
            no_runs[scen_name] += 1
            total_runs += 1

        # calculate success rates
        succ_rates: Dict[str, float] = {
            scen: no_successes[scen] / no_runs[scen] for scen in scenarios
        }

        # evaluate if the thresholds are passed
        hard_passed, soft_passed, thresh_success = self.evaluate_threshs(succ_rates)
        # print results to console
        self.logger.info(
            f"Ran {len(results)} tests; Average success rate {round(total_successes/total_runs*100,2)}%"
        )
        self.show_test_results(
            no_runs, no_successes, succ_rates, thresh_success, hard_passed, soft_passed
        )

        # raise an error if the tests have failed
        if not hard_passed or not soft_passed:
            self.fail_tests(hard_passed, soft_passed)

    def evaluate_threshs(
        self, succ_rates: Dict[str, float]
    ) -> Tuple[bool, bool, Dict[str, Dict[str, bool]]]:
        """Evaluates the evaluation results against the test thresholds.

        The method evaluates the success rates of each individual scenario against
        the corresponding hard and soft thresholds specified in the test_thresholds.json.

        Parameters
        ----------
        succ_rates : Dict[str, float]
            Dictionary with scenario names as keys and success rates as values.

        Returns
        -------
        hard_passed : bool
            Flag indicating whether all hard thresholds have been passed.
        soft_passed : bool
            Flag indicating whether all soft thresholds have been passed.
        thresh_passed_all : Dict[str, Dict[str, bool]]
            Nested dictionary containing threshold information for each individual scenario.
        """
        thresh_passed_all: Dict[Dict[str, bool]] = {}
        for scenario, succ_rate in succ_rates.items():
            # get hard and soft thresholds from config
            hard, soft = (
                self.thresholds[scenario]["hard"],
                self.thresholds[scenario]["soft"],
            )
            # Evaluate thresholds against achieved success rate and store
            scenario_eval = {
                "hard_passed": hard <= succ_rate,
                "soft_passed": soft <= succ_rate,
            }
            thresh_passed_all[scenario] = scenario_eval

        # set test success flags for later use
        hard_passed = all([r["hard_passed"] for r in thresh_passed_all.values()])
        soft_passed = all([r["soft_passed"] for r in thresh_passed_all.values()])

        return hard_passed, soft_passed, thresh_passed_all

    @staticmethod
    def extract_run_info(run: Dict[str, any]) -> Tuple[str, int]:
        """Gets relevant information from a scenario result.

        The method extracts the name of the evaluated scenario together with the success rate.
        A scenario is deemed a success if the following results are avoided:
        - Car collided
        - Car invalid

        Parameters
        ----------
        run : List[Any]
            List containing the result information for a single scenario.

        Returns
        -------
        str
            Name of the scenario which has been evaluated.
        int
            0 if the scenario has not been completed successfully, 1 otherwise.
        """
        scenario_name = run["scenario"]
        if run["carsCollided"] or run["carsInvalid"]:
            return scenario_name.lower(), 0
        else:
            return scenario_name.lower(), 1

    def print_test_settings(self) -> None:
        """Print a tabular overview over the tests to be performed.

        The table shows the following information for each scenario:
            1. Soft threshold.
            2. Hard threshold.
            3. Number of runs for this scenario.
            4. Number of MCTS iterations for this scenario.
        """

        self.logger.info(f"\nTEST CONFIGURATION:\n")
        # set up rows and columns
        col_names = ("soft_thresh", "hard_thresh", "iterations", "runs")
        row_names = self.bulk_config["scenarios"]
        row_format = "{0:>5}"
        # set the columns spacing
        for i in range(1, len(col_names) + 1):
            row_format += "{" + str(i) + ":>15}"
        # print the column headers
        self.logger.info(row_format.format("", *col_names))

        # write the results into the table
        for scenario in self.bulk_config["scenarios"]:
            self.logger.info(
                row_format.format(
                    scenario,
                    f"{self.thresholds[scenario]['soft']:.2f}",
                    f"{self.thresholds[scenario]['hard']:.2f}",
                    self.bulk_config["options_alterations"]["compute_options"][
                        "n_iterations"
                    ][0],
                    self.bulk_config["number_runs"],
                )
            )
        self.logger.info(f"Total number of test runs: {self.number_evaluations}.")

    def show_test_results(
        self,
        no_runs: Dict[str, int],
        no_successes: Dict[str, int],
        succ_rates: Dict[str, float],
        thresh_passed: Dict[str, Dict[str, bool]],
        hard_passed: bool,
        soft_passed: bool,
    ) -> None:
        """Prints the evaluation results as a table.

        The table shows the success rate for each scenario along with the hard and soft thresholds.
        For each threshold level an indicator shows whether it has been passed or not.

        Parameters
        ----------
        succ_rates : Dict[str, float]
            Dictionary holding the success rates for each scenario as values with the scenarios as keys.
        thresh_passed: Dict[str, Dict[str, bool]]
            Nested Dictionary storing information whether each scenario has passed the tests.
            For each scenario a dictionary with a boolean flag for passing hard and soft thresholds is held.
        hard_passed: bool
            Boolean flag for passing hard thresholds.
            True if all hard thresholds have been passed, False once a single threshold is failed.
        soft_passed: bool
            Boolean flag for passing soft thresholds.
            True if all soft thresholds have been passed, False once a single threshold is failed.
        """
        # Print colored header depending on success
        if not hard_passed:
            self.logger.error(f"{self.c_error} TESTS FAILED. {self.c_reset}")
        elif not soft_passed:
            self.logger.warning(
                f"{self.c_warning} TESTS passed with warnings.{self.c_reset}"
            )
        else:
            self.logger.info(f"{self.c_okay} ALL TEST PASSED.{self.c_reset}")

        # define row and column names
        col_names = (
            "runs",
            "successes",
            "avg_success",
            "soft_thresh",
            "soft_passed",
            "hard_thresh",
            "hard_passed",
        )
        row_names = thresh_passed.keys()
        # define the row width
        row_format = "{0:>5}"
        for i in range(1, len(col_names) + 1):
            row_format += "{" + str(i) + ":>15}"
        # print the columns headers
        self.logger.info(row_format.format("", *col_names))

        # unpack results and print each row
        # row color depends on the test success
        for scenario, vals in thresh_passed.items():
            hard_passed = vals["hard_passed"]
            soft_passed = vals["soft_passed"]
            succ_rate = succ_rates[scenario]
            row_string = row_format.format(
                scenario,
                no_runs[scenario],
                no_successes[scenario],
                f"{succ_rate:.2f}",
                f"{self.thresholds[scenario]['soft']:.2f}",
                str(soft_passed),
                f"{self.thresholds[scenario]['hard']:.2f}",
                str(hard_passed),
            )
            if not hard_passed:
                self.logger.error(self.c_error + row_string + self.c_reset)
            elif not soft_passed:
                self.logger.warning(self.c_warning + row_string + self.c_reset)
            else:
                self.logger.info(self.c_okay + row_string + self.c_reset)

    def fail_tests(self, hard_passed: bool, soft_passed: bool) -> None:
        """Method for failing the CI tests if necessary.

        Raises a TestFailedError if the hard thresholds are not met.
        Prints a warning if the hard thresholds are passed but soft thresholds are failed.
        The CI pipeline is only failed if hard thresholds are missed.

        Parameters
        ----------
        hard_passed: bool
            Boolean flag indicating if all hard thresholds have been passed.
        soft_passed: bool
            Boolean flag indicating if all soft thresholds have been passed.
        """
        if not hard_passed:
            raise TestFailedError(f"{self.c_error} TESTS FAILED.{self.c_reset}")
        elif not soft_passed:
            warnings.showwarning(
                f"{self.c_warning} TESTS passed with warnings. Only soft thresholds passed.{self.c_reset}",
                category=RuntimeWarning,
                filename="tests",
                lineno=352,
            )


if __name__ == "__main__":
    """
    Main for testing multiple evaluator configurations in sequence.
    This code is called in the gitlab-ci pipeline in both base-library
    and ros-executable (`proseco_planning` and `ros_proseco_planning`).

    It detects all evaluator configuration json files in a path and tries
    to load the success rate threshold values from a json file that
    prepends the string `threshold_<filename>.json`.

    If the evaluator returns any success rate for a configuration lower than
    the hard threshold, then the test is regarded as a failure.
    A soft threshold only gives a warning.
    """
    parser = argparse.ArgumentParser(
        description="Test multiple configurations using the ProSeCo evaluator. "
        "A minimum success rate is required to be successfull."
    )
    parser.add_argument(
        "-p", "--path", help="Specify a path that contains test `json` files."
    )
    parser.add_argument(
        "-s",
        "--no_summary",
        action="store_true",
        default=False,
        help="specifies whether to create a results.json file in the root evaluation folder",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="specifies whether the MCTS output is printed to the console",
    )
    parser.add_argument("--address", help="Override the ray cluster address")
    args = parser.parse_args()

    # Validate path
    pack_path = rospkg.RosPack().get_path("ros_proseco_planning")
    default_path = os.path.join(pack_path, "config/evaluator/ci_config")
    path = os.path.abspath(
        args.path or default_path
    )  # use argument if set, else default path
    assert os.path.isdir(path)

    # Find configs
    json_files = [
        os.path.basename(f)
        for f in glob.iglob(os.path.join(path, "*.json"))
        if os.path.isfile(f)
    ]
    json_files = [f for f in json_files if not f.startswith("threshold_")]

    configs = {
        f: f"threshold_{f}"
        if os.path.isfile(os.path.join(path, f"threshold_{f}"))
        else default_thresholds
        for f in json_files
    }

    # Setup logger
    logger = get_logger("ProSeCo Tester", logging.INFO)

    # Run tests
    logger.info(f"Found configurations: {list(configs.keys())}")

    for n, (config, threshold) in enumerate(configs.items()):
        # Construct arguments for evaluator
        test_args = argparse.Namespace()
        test_args.config = os.path.abspath(os.path.join(path, config))
        test_args.yes = True
        test_args.no_dashboard = True
        test_args.no_summary = args.no_summary
        test_args.debug = args.debug
        test_args.address = args.address

        # Validate files
        assert os.path.isfile(test_args.config)
        threshold_file = os.path.abspath(os.path.join(path, threshold))
        assert os.path.isfile(threshold_file)

        # Run tests
        logger.info("=" * 60)
        logger.info(f'Testing "{config}" with thresholds "{threshold}"')
        tester = ProSeCo_Tester(
            test_args, init_ray=(n == 0), threshold_file=threshold_file, logger=logger
        )
        tester.run_tests()
