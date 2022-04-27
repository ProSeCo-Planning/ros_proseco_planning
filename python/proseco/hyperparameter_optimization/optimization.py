import argparse
from pathlib import Path
import logging

# Evaluator
from proseco.hyperparameter_optimization.analyzer import Analyzer
from proseco.hyperparameter_optimization.optimizer import Optimizer
from proseco.utility.io import add_extension, get_absolute_path, load_data
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)  # debug, info, warning, error, critical


def validate_config(config: Dict[str, Any]) -> None:
    """Validates the configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        The configuration dictionary.
    """
    assert len(config["optimizer"]["multicriterial_weights"]) == len(
        config["optimizer"]["optimization_function"]
    ), "The length of the weights list must be equal to the length of the optimization function list."

    assert config["optimizer"]["type"] in [
        "smac_4HPO",
        "smac_4BB",
        "random",
        "same_options",
    ], "The optimizer type is not unsupported."


def parse_arguments():
    """Load command line arguments for the evalution.

    Takes the -p, -c and -f arguments.
    Returns the openend program file extended by the configspace json dictionary.

    Returns
    -------
    Dict[str, Any]
        Opened program json file extended by the configspace dictionary.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--configuration",
        help="Configuration file for the analyzer, optimizer or optimizer_comparison",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--function",
        help="The function that should be used, i.e. analyzer, optimizer or optimizer_comparison",
        type=str,
    )
    parser.add_argument("-d", "--debug", help="Enable debug mode", action="store_true")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()

    config_path = get_absolute_path(
        Path("config/hyperparameter_optimization")
        / add_extension(args.configuration, "json")
    )
    config = load_data(config_path)
    validate_config(config)

    optimizer = Optimizer(config, args.debug)
    analyzer = Analyzer(config, optimizer)
    # Optimization
    if args.function == "optimizer":
        X, y, t, y_d = optimizer.optimize()
        analyzer.plot_trajectory(y, t, optimizer.args["type"])
    # Compare different optimization algorithms
    elif args.function == "optimizer_comparison":
        comparison_dict, significance_dict = analyzer.compare_algorithms(
            config, optimizer
        )
    # Analyzes evaluator output files
    elif args.function == "analyzer":
        analyzed_results = analyzer.open_evaluation()
        analyzer.plot_result_dict(analyzed_results)
    else:
        logging.error(
            "Please specify which function of the optimizer you"
            + " want to use: analyzer, optimizer or optimizer_comparison."
            + " e.g., type -f optimizer"
        )
