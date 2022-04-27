#!/usr/bin/env python3

"""
Created 05.08.2020
@authors: Karl Kurzer

Script to generate JSON scenario configurations from CUE files.
"""

from pathlib import Path
import subprocess

from proseco.utility.ui import get_logger

logger = get_logger("ProSeCo Scenario Generator")


def format_cue_scenarios(cue_dir: Path) -> None:
    """
    Format all CUE scenario files.

    Notes
    -----
    This requires the CUE executable. https://cuelang.org/

    Parameters
    ----------
    cue_dir : Path
        directory in which the CUE scenario files are located
    """
    for cue_scenario in cue_dir.glob("sc[0-9]*.cue"):
        cmd = f"cue fmt {cue_scenario}"
        logger.info(cmd)
        subprocess.run(cmd, shell=True, check=True)
    logger.info("Formatted all cue scenarios")


def validate_cue_scenarios(cue_dir: Path) -> None:
    """
    Validate all CUE scenario files.

    Notes
    -----
    This requires the CUE executable. https://cuelang.org/

    Parameters
    ----------
    cue_dir : Path
        directory in which the CUE scenario files are located
    """
    for cue_scenario in cue_dir.glob("sc[0-9]*.cue"):
        cmd = f"cue vet {cue_scenario}"
        logger.info(cmd)
        subprocess.run(cmd, shell=True, check=True)
    logger.info("Validated all cue scenarios")


def create_json_scenarios(cue_dir: Path, json_dir: Path) -> None:
    """
    Create all JSON scenario files based on the CUE definitions.

    Notes
    -----
    This requires the CUE executable. https://cuelang.org/

    Parameters
    ----------
    cue_dir : Path
        directory in which the CUE scenario files are located
    json_dir : Path
        directory in which the JSON scenario files are exported
    """
    for cue_scenario in cue_dir.glob("sc[0-9]*.cue"):
        # path to the new json file
        json_scenario = json_dir / (cue_scenario.stem + ".json")
        cmd = f"cue export -f {cue_scenario} -o {json_scenario}"
        logger.info(cmd)
        subprocess.run(cmd, shell=True, check=True)
    logger.info("Created all json scenarios")


def validate_json_scenarios(json_dir: Path, cue_dir: Path) -> None:
    """
    Validate all CUE scenario files based on the CUE definitions.

    Notes
    -----
    This requires the CUE executable. https://cuelang.org/

    Parameters
    ----------
    json_dir : Path
        directory in which the JSON scenario files are located
    cue_dir : Path
        directory in which the CUE scenario files are located
    """
    for json_scenario in json_dir.glob("sc[0-9]*.json"):
        cue_scenario = cue_dir / (json_scenario.stem + ".cue")
        if not cue_scenario.exists():
            logger.warning(f"The file {cue_scenario} doesn't exist")
            continue
        cmd = f"cue vet {json_scenario} {cue_scenario}"
        logger.info(cmd)
        subprocess.run(cmd, shell=True, check=True)
    logger.info("Validated all json scenarios")


if __name__ == "__main__":
    # path of this file
    path = Path(__file__)
    # path of the directory
    file_dir = path.parent

    # some steps may consume a lot of time, so feel free to comment out
    format_cue_scenarios(file_dir)
    # validation is not explicitly necessary because this ìs also done during the export in `create_json_scenarios`
    # validate_cue_scenarios(file_dir)
    create_json_scenarios(file_dir, file_dir)

    # json scenarios can also be validated, but that's not done here because they have just been created according to the cue defintions
    # validate_json_scenarios(file_dir, file_dir)
