import os
import copy
import ray
import argparse
import tqdm
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil
import uuid as uu_id
from typing import ClassVar, Dict, Generator, Tuple, List, Any

from proseco.evaluator.remote import run
from proseco.evaluator.progressBar import ProgressBar
import proseco.utility.io as io
import proseco.utility.ui as ui
from proseco.dashboard.model import (
    get_run_directories,
    is_successful_result,
    load_result,
    load_options,
    load_scenario,
    load_uuid,
)
from proseco.evaluator.util import (
    flatten_dictionary,
    nest_dictionary,
    hash_dict,
    get_permutations,
)


class Head:
    """Head node of the Evaluation."""

    pack_path: str
    args: argparse.Namespace
    bulk_config: Dict[str, Any]
    time_string: str
    number_evaluations: int
    iteration_index: int
    logger: logging.Logger
    c_error: ClassVar[str] = "\033[91m\033[1m"
    c_warning: ClassVar[str] = "\33[33m\033["
    c_okay: ClassVar[str] = "\033[92m\033[1m"
    c_reset: ClassVar[str] = "\033[0m"

    def __init__(
        self, args: argparse.Namespace, time_string: str = None, init_ray: bool = True
    ):
        """Constructor.

        Parameters
        ----------
        args : argparse.Namespace
            Command line arguments for starting the evaluation.
        time_string : str
            Optional time stamp when this head node was started.
        init_ray : bool
            When True the evaluation will connect the local ray client to a cluster (default: True).
            It is used to prevent multiple ray-inits when running evaluations in sequence, else ray init will throw an exception.
        """
        self.logger = ui.get_logger("ProSeCo Evaluator")
        # Arguments with flags
        self.args = args
        # Openend config directory
        self.bulk_config = self.get_file(file_type="evaluator", file=args.config)
        # Timestamp of the head node
        if time_string == None:
            self.time_string = io.create_timestamp()
        else:
            self.time_string = time_string
        # How many evaluations this config will generate
        self.number_evaluations = self.determine_number_evaluations()
        # The current iteration index the evaluation is at
        self.iteration_index = 0
        self.path = self.create_evaluation_directory()
        # The time in seconds, when a task is considered to be timed out.
        self.task_timeout = 120

        self.isolate_binary()
        self.init_ray()

    def __del__(self):
        """Destructor.

        Removes the temporary binary for the evaluation.
        """
        self.remove_binary()

    def determine_number_evaluations(self) -> int:
        """Determines the number of MCTS-Evaluations to be started.

        Returns
        -------
        int
            Total number of scenario evaluations.
        """
        len_options = len(self.bulk_config["options"])
        len_scenarios = len(self.bulk_config["scenarios"])
        number_runs = self.bulk_config["number_runs"]

        alterations = flatten_dictionary(self.bulk_config["options_alterations"])
        scenario_alterations = flatten_dictionary(
            self.bulk_config["scenario_alterations"]
        )
        alterations.update(scenario_alterations)

        len_alter = 1
        for _, v in alterations.items():
            if type(v) is list:
                len_alter = len_alter * len(v)
            else:
                pass

        number_evaluations = len_options * len_scenarios * len_alter * number_runs

        return number_evaluations

    def create_evaluation_directory(self) -> Path:
        """Creates a directory for the evaluation.

        Returns
        -------
        Path
            The path to the evaluation directory.
        """
        path = (
            io.get_user_temp_dir()
            / f"proseco_evaluator_output/{self.time_string}_{self.bulk_config['evaluation_name']}"
        )
        path.mkdir(parents=True)
        self.logger.info(f"Creating evaluation directory {path}")
        return path

    def isolate_binary(self) -> None:
        """Moves the binary to a unique directory, so that rebuilding the binary does not affect the evaluation."""

        devel_binary_path = (Path(io.get_ros_pack_path()).parents[1]).joinpath(
            "devel_isolated/ros_proseco_planning/lib/ros_proseco_planning"
        )
        temp_binary_path = Path(f"~/evaluator_bin_{self.time_string}").expanduser()
        shutil.copytree(devel_binary_path, temp_binary_path)
        self.logger.debug(f"Created temporary binary path {temp_binary_path}")
        self.binary_path = temp_binary_path

    def remove_binary(self) -> None:
        """Removes the temporary binary for the evaluation."""
        self.logger.debug(f"Removed temporary binary path {self.binary_path}")
        shutil.rmtree(self.binary_path)

    def init_ray(self) -> None:
        """Initializes the ray cluster, determines the maximum number of workers and creates the workers."""
        self.logger.debug(f"Initializing ray cluster")
        ray.init(
            address=self.args.address or self.bulk_config["ray_cluster"]["address"],
            include_dashboard=not self.args.no_dashboard,
            dashboard_host="0.0.0.0",
            log_to_driver=self.args.debug,
            _temp_dir=str(io.get_user_temp_dir() / "ray"),
        )

    def print_evaluation_scenarios(self) -> None:
        """Prints the different combinations of settings for the evaluation."""
        self.logger.debug(f"EVALUATOR CONFIGURATION: {self.bulk_config}")
        scenarios = ",".join(self.bulk_config["scenarios"])
        self.logger.info(f"Evaluating Scenarios: [{scenarios}]")

    @staticmethod
    def permute_dictionary(
        dictionary: Dict[str, any], permutation: Dict[str, any]
    ) -> Dict[str, any]:
        """Creates a new dictionary with the given permutation applied.

        Parameters
        ----------
        dictionary : Dict[str, any]
            The dictionary to apply the permutation to.
        permutation : Dict[str, any]
            The permutation to apply.

        Returns
        -------
        Dict[str, any]
            The new options dictionary.
        """
        dictionary = flatten_dictionary(dictionary)
        dictionary.update(permutation)
        return nest_dictionary(dictionary)

    @staticmethod
    def remove_random_seed(options: Dict[str, any]) -> Dict[str, any]:
        """Removes the random seed from the options.

        Parameters
        ----------
        options : Dict[str, any]
            The options to remove the random seed from.

        Returns
        -------
        Dict[str, any]
            The options without the random seed.
        """
        options = copy.deepcopy(options)
        options["compute_options"].pop("random_seed")
        return options

    @staticmethod
    def hash_options_and_scenario(
        options: Dict[str, Any], scenario: Dict[str, Any]
    ) -> str:
        """Hashes the options and the scenario.

        Parameters
        ----------
        options : Dict[str, Any]
            The options to hash.
        scenario : Dict[str, Any]
            The scenario to hash.

        Returns
        -------
        str
            The combination of the options and the scenario.
        """
        options_scenario = {}
        options_scenario.update(options)
        options_scenario.update(scenario)
        return hash_dict(options_scenario)

    def get_file(self, file_type: str, file: str) -> Dict[str, Any]:
        """Returns a scenario, config or options dictionary.

        The files are loaded via json.

        Parameters
        ----------
        file_type : str
            String indicating whether the file to load contains options, a scenario or a config for the evaluator.
        file  : str
            Name of the file to load.

        Returns
        -------
        Dict[str, Any]
            Loaded options dictionary.
        """
        if isinstance(file, dict):
            data = file

        elif type(file) is str:

            if not file.endswith(".json"):
                file += ".json"

            if (
                file_type == "options"
                or file_type == "scenarios"
                or file_type == "evaluator"
            ):
                path = io.get_ros_pack_path() / "config" / file_type / file
            else:
                raise Exception(f"Unknown file type {file_type}")

            data = io.load_data(path)

        return data

    def options_iterator(
        self,
    ) -> Generator[
        Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], List[str], Dict[str, Any]],
        None,
        None,
    ]:
        """Defines an iterator for the option permutations.

        The iterator returns the necessary files for evaluating a scenario with a worker node.
        Additionally, it also returns an info dict used to update the progress bar.

        Yields
        -------
        options: Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], List[str], Dict[str, Any]]
            Tuple containing all the necessary files to initiate an evaluation run. Contains:
                - new_options : Dictionary containing compute options.
                - new_scenario : Loaded scenario configuration.
                - uuid : Unique IDs and scenario-options hashes.
                - info_dict: Information to update the progress bar.
        """

        # Iterate over all provided compute option files
        for options in self.bulk_config["options"]:
            options = io.load_options(options)

            flat_options = flatten_dictionary(self.bulk_config["options_alterations"])
            options_permutations = get_permutations(flat_options)

            # Iterate over all permutations of options alterations
            for options_permutation in options_permutations:
                self.logger.debug(f"Permuting options with: {options_permutation}")
                new_options = self.permute_dictionary(options, options_permutation)

                # Unique ID for the options file
                uuid_options = str(uu_id.uuid4())
                new_options_wo_random_seed = self.remove_random_seed(new_options)
                options_hash = hash_dict(new_options_wo_random_seed)

                # Iterate over every scenario
                for scenario in self.bulk_config["scenarios"]:
                    scenario = io.load_scenario(scenario)

                    flat_scenario = flatten_dictionary(
                        self.bulk_config["scenario_alterations"]
                    )
                    scenarios_permutations = get_permutations(flat_scenario)

                    # Iterate over all scenario permutations of the scenario alterations
                    for scenarios_permutation in scenarios_permutations:
                        self.logger.debug(
                            f"Permuting scenario with: {scenarios_permutation}"
                        )
                        new_scenario = self.permute_dictionary(
                            scenario, scenarios_permutation
                        )

                        # Unique ID for the options and scenario tuple
                        uuid_scenario = str(uu_id.uuid4())
                        options_scenario_hash = self.hash_options_and_scenario(
                            new_options_wo_random_seed, new_scenario
                        )

                        uuids = {
                            "options_uuid": uuid_options,
                            "options_scenario_uuid": uuid_scenario,
                            "options_hash": options_hash,
                            "options_scenario_hash": options_scenario_hash,
                        }

                        # Iterate over number of runs
                        for _ in range(self.bulk_config["number_runs"]):
                            yield new_options, new_scenario, uuids

    def run_tasks(self):
        """Starts the tasks and waits for them to finish.

        Parameters
        ----------
        results : List[List[Any]]
            The list of results.
        """
        pb = ProgressBar(self.number_evaluations)

        results_ref = []
        for options, scenario, uuids in self.options_iterator():
            results_ref.append(
                run.remote(
                    options,
                    scenario,
                    uuids,
                    self.binary_path,
                    self.args.debug,
                    pb.actor,
                )
            )
        pb.print_until_done_or_timeout(self.task_timeout)
        results = ray.get(results_ref)
        return results

    def start(self) -> Tuple[List[List[Any]], bool]:
        """Creates all the ray futures according to the configuration file and evaluates them.

        Main method for running the evaluation and delegating tasks to workers.

        Notes
        -----
        If the result list becomes larger than 5GB, the result list returned by this method does not contain all results.
        The return value `complete` indicates whether all results are contained.
        Despite the limitation of the returned list, all results are always persisted to disk in the evaluation directory.

        Returns
        -------
        Tuple[List[List[Any]], bool]
            Tuple containing the results of the evaluation.
            The first element is a list with the current results.
            The second element is a boolean flag indicating whether the list contains ALL results.
        """
        self.print_evaluation_scenarios()
        # Starting time of the evaluation
        beginning = datetime.now()

        results = self.run_tasks()
        self.save_results(results)

        # create summary json if specified
        if not self.args.no_summary:
            _ = self.create_summary(True)

        self.logger.info(
            f"{self.c_okay}The {self.iteration_index}-th evaluation terminated successfully in {datetime.now()-beginning}{self.c_reset}"
        )
        self.iteration_index += 1

        return results

    def save_results(self, results: List[List[Any]]) -> None:
        """Saves the results to disk.

        Parameters
        ----------
        results : List[List[Any]]
            The list containing the results of all runs.
        """

        for result_index in range(len(results)):
            self.save_result(results[result_index], result_index)

    def save_result(self, result: List[Any], result_index: int) -> None:
        """Saves a single result to disk.

        Parameters
        ----------
        result : List[Any]
            The list containing the partial results of a single run.
        result_index : int
            The index of the result.
        """
        result_path = (
            self.path
            / f"iteration_{self.iteration_index}"
            / f"result_{str(result_index).zfill(io.get_number_digits(self.number_evaluations-1))}of{str(self.number_evaluations-1).zfill(io.get_number_digits(self.number_evaluations-1))}"
        )
        result_path.mkdir(parents=True)

        for file_name, data in result:
            io.save_data(data, result_path / file_name)

        # append the path to the result
        result.append(["%PATH%", str(result_path)])

    @staticmethod
    def _check_alterations_key(key: str, value: Any) -> bool:
        """Checks whether the specified key is being iterated over in the evaluator bulk config.

        Parameters
        ----------
        key : str
            String dictionary key of an options file.
        value : Any
            Dictionary value.

        Returns
        -------
        bool
            True if the key is a compute_options alteration, False else.
        """
        return (
            isinstance(value, list)
            and key.startswith("options_alterations")
            and not key.endswith("seed")
            and not key.split("/")[1] == "output_options"
        )

    def create_summary(self, save_to_disk: bool = False) -> List[Dict]:
        """Returns a summary of the results and saves it in the evaluation directory if the flag argument is True.

        Parameters
        ----------
        save_to_disk : bool = False
            Flag for saving the summary to disk.

        Returns
        -------
        List[Dict]
            List containing summary dicts for each run. The summary contains:
            Scenario name, options_uuid, path to the results folder, success flag.
        """
        self.logger.info("Summarizing results")
        results = []

        # Get the config keys that define alterations from the bulk config
        co_keys = [
            key.lstrip("options_alterations/")
            for key, value in (flatten_dictionary(self.bulk_config)).items()
            if self._check_alterations_key(key, value)
        ]

        for run_dir in get_run_directories(self.path):
            data = load_result(run_dir)
            options = load_options(run_dir)
            scenario = load_scenario(run_dir)
            uuid = load_uuid(run_dir)

            data["scenario"] = scenario["name"]
            # Add uuids and hashes to the results file
            data.update(uuid)
            # added the path to the results so that the folder can be found for the visualization
            data["path"] = str(run_dir.resolve())

            # Load the altered options from the options file and append them to the results
            if co_keys:
                flat_opt = flatten_dictionary(options)
                for key in co_keys:
                    name = key.split("/")[-1]
                    data[name] = flat_opt[key]

            # determine whether the run was successful or not
            data["success"] = is_successful_result(data)

            results.append(data)

        # persist summary to disk
        if save_to_disk:
            path = self.path / "results.json"
            io.save_data(results, path)
        return results

    def create_result_dataframe(self, save_to_disk: bool = False) -> pd.DataFrame:
        """Returns detailed result information including the compute options and saves them in the evaluation directory if the flag argument is True.

        Parameters
        ----------
        save_to_disk : bool = False
            Flag for saving the summary to disk.

        Returns
        -------
        pandas.DataFrame
            DataFrame generated from the individual run results. Contains for each run:
            Complete options, detailed result, uuid, scenario name.
        """
        all_data = []

        for result_path, options_path, scenario_path, uuid_path in tqdm(
            zip(
                self.path.rglob("result.json"),
                self.path.rglob("options_output.json"),
                self.path.rglob("scenario_output.json"),
                self.path.rglob("uuid.json"),
            ),
            total=self.number_evaluations,
            ascii=True,
            desc="Generating DataFrame",
        ):
            # open all files and extract the information
            options = pd.json_normalize(io.load_data(options_path))
            results = pd.json_normalize(io.load_data(result_path))
            uuid = pd.json_normalize(io.load_data(uuid_path))

            run_data = pd.concat([options, uuid, results], axis=1)
            run_data["scenario"] = io.load_data(scenario_path)["name"]
            all_data.append(run_data)

        df_all = pd.concat(all_data, axis=0)
        df_all.reset_index(drop=True, inplace=True)

        if save_to_disk:
            df_all.to_pickle(self.path / "run_data.gz", compression="gzip")

        return df_all
