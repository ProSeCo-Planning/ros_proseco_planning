#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import getpass
import json
import logging
from pathlib import Path
import shutil
from typing import cast, Optional

from proseco.dashboard.model import (
    load_data,
    load_result,
    load_summary,
    get_run_directories,
)

# default path to the IRL directory
g_irl_path = Path(f"/tmp/{getpass.getuser()}/irl")
# default path to the IRL experts directory
g_irl_experts_path = g_irl_path / "experts"


def load_irl_trajectory(run_path: Path) -> dict:
    """Return the IRL trajectory as a dictionary.

    Parameters
    ----------
    run_path : Path
        Path to the run directory

    Returns
    -------
    dict
        dictionary of the IRL trajectory
    """
    # assumes that there exists only one corresponding file
    irl_trajectory_path = next(run_path.glob("irl_trajectory.*"))
    irl_trajectory = cast(dict, load_data(irl_trajectory_path))
    return irl_trajectory


def uses_invalid_action(irl_trajectory: dict) -> bool:
    """Return true if an invalid action is used in the IRL trajectory.

    Parameters
    ----------
    irl_trajectory : dict
        dictionary of the IRL trajectory

    Returns
    -------
    bool
        true if an invalid actions is used
    """
    for agent_dict in irl_trajectory["agents"]:
        for trajectory_step in agent_dict["trajectory"]:
            if trajectory_step["features"]["invalidAction"]:
                return True
    return False


def remove_unsuccessful_runs(
    args: argparse.Namespace,
    *,
    b_collided: bool = True,
    b_invalid_state: bool = True,
    b_desire: bool = True,
    b_invalid_action: bool = True,
) -> None:
    """Remove all unsuccessful runs.

    Parameters
    ----------
    args : argparse.Namespace
        arguments
    b_collided : bool, optional
        if true, runs with collisions are removed, by default True
    b_invalid_state : bool, optional
        if true, runs with invalid states are removed, by default True
    b_desire : bool, optional
        if true, runs with unfulfilled desires are removed, by default True
    b_invalid_action : bool, optional
        if true, runs with invalid actions are removed, by default True
    """
    eval_path: Path = args.eval_path

    for run_path in get_run_directories(eval_path):
        result: dict = load_result(run_path)
        irl_trajectory = load_irl_trajectory(run_path)
        if (
            (b_collided and result["carsCollided"])
            or (b_invalid_state and result["carsInvalid"])
            or (b_desire and not result["desiresFulfilled"])
            or (b_invalid_action and uses_invalid_action(irl_trajectory))
        ):
            _remove_run(run_path, args.irl_experts_path, result)


def get_irl_message_path(
    run_path: Path, irl_experts_path: Path, result: Optional[dict] = None
) -> Path:
    """Get the path to the IRL expert message pickle file for a specific run.

    Parameters
    ----------
    run_path : Path
        Path to the run directory
    irl_experts_path : Path
        Path to the IRL experts directory
    result : Optional[dict], optional
        result of the run, by default None

    Returns
    -------
    Path
        Path to the IRL message pickle file
    """
    if result is None:
        # load the result
        result: dict = load_result(run_path)  # type: ignore [no-redef]
    result = cast(dict, result)

    scenario: str = result["scenario"]
    run_num = get_run_number(run_path)
    irl_message_path = (
        irl_experts_path
        / "pickle_files"
        / scenario.lower()
        / f"expert_message_{run_num}.p"
    )
    return irl_message_path


def get_run_number(run_path: Path) -> int:
    """Get the "run number" of a specific run.

    Parameters
    ----------
    run_path : Path
        Path to the run directory

    Returns
    -------
    int
        "run number"
    """
    run_info = run_path.name.split("_")[3]
    run_num_str = run_info.split("of")[0]
    run_num = int(run_num_str)
    return run_num


def remove_run(args: argparse.Namespace) -> None:
    """Remove a specific run.

    Parameters
    ----------
    args : argparse.Namespace
        arguments
    """
    _remove_run(args.run_path, args.irl_experts_path)


def _remove_run(
    run_path: Path, irl_experts_path: Path, result: Optional[dict] = None
) -> None:
    """Remove a specific run.

    Parameters
    ----------
    run_path : Path
        Path to the run directory
    irl_experts_path : Path
        Path to the IRL experts directory
    result : Optional[dict], optional
        result of the run, by default None
    """
    irl_message_path = get_irl_message_path(run_path, irl_experts_path, result)

    irl_message_path.unlink()
    logger.info(f"Removed {irl_message_path}")
    shutil.rmtree(run_path)
    logger.info(f"Removed {run_path}")
    _remove_run_from_summary(run_path)


def _remove_run_from_summary(run_path: Path) -> None:
    """Remove a specific run from the summary json file.

    Parameters
    ----------
    run_path : Path
        Path to the run directory
    """
    eval_dir = run_path.parents[1]
    summary = load_summary(eval_dir)

    for i, result in enumerate(summary):
        path_str_list = result["path"].rsplit("/", maxsplit=2)
        path_str = path_str_list[1] + "/" + path_str_list[2]

        if str(run_path.resolve()).endswith(path_str):
            del summary[i]
            with open(eval_dir / "results.json", "w") as f:
                json.dump(summary, f, indent=4)
            logger.info(f"Removed {run_path} from summary")
            break


def remove_runs_above_upper_bound(args: argparse.Namespace) -> None:
    """Remove all runs whose "run number" is larger than the `args.upper_bound`.

    Parameters
    ----------
    args : argparse.Namespace
        arguments
    """
    eval_path: Path = args.eval_path

    for run_path in get_run_directories(eval_path):
        run_num = get_run_number(run_path)
        if run_num > args.upper_bound:
            _remove_run(run_path, args.irl_experts_path)


def adjust_run_path(args: argparse.Namespace) -> argparse.Namespace:
    """Adjust the run path so that this path is a subpath of the evaluation path.

    Parameters
    ----------
    args : argparse.Namespace
        arguments

    Returns
    -------
    argparse.Namespace
        arguments with an adjusted run path
    """
    eval_path = cast(Path, args.eval_path).resolve()
    run_path = cast(Path, args.run_path).resolve()

    if run_path.parents[1] == eval_path:
        # run_path is subpath of eval_path, so no adjustment needed
        return args

    args.run_path = eval_path / run_path.parts[-2] / run_path.parts[-1]
    return args


def parse_args() -> argparse.Namespace:
    """Parse the arguments for this script.

    Returns
    -------
    argparse.Namespace
        the arguments for this script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--eval_path", type=Path, help="Path to the evaluation directory"
    )
    parser.add_argument("-r", "--run_path", type=Path, help="Path to the run directory")
    parser.add_argument(
        "-i",
        "--irl_experts_path",
        type=Path,
        default=g_irl_experts_path,
        help="Path to the IRL experts directory",
    )
    parser.add_argument(
        "-u",
        "--upper_bound",
        type=int,
        help="Integer that specifies the upper bound for the run number.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("filter_experts")
    args = parse_args()

    if args.run_path:
        # remove a specific run
        if args.eval_path:
            # adjust the run path so that this path is a subpath of the evaluation path
            args = adjust_run_path(args)
        remove_run(args)
    elif args.eval_path:
        # remove several runs from an evaluation
        if args.upper_bound is not None:
            remove_runs_above_upper_bound(args)
        else:
            remove_unsuccessful_runs(args, b_desire=True)
    else:
        logger.error("Either --eval_path or --run_path must be specified")
