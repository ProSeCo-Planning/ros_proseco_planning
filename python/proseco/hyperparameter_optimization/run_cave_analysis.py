from pathlib import Path
import subprocess
from proseco.utility.ui import (
    let_user_select_subdirectory,
    start_web_server,
    get_logger,
)


def run_cave_analysis(output_path: Path) -> None:
    """Analyzes the output of the SMAC optimizer and generates CAVE output files.

    Parameters
    ----------
    output_path : Path
        The path to the directory, where the optimizer output files are stored.
    """

    cave_output_path = output_path / "cave"
    smac_output_path = output_path / "smac"

    run_directories = list(smac_output_path.iterdir())
    assert len(run_directories) == 1, "The smac directory contains more than one run."
    run_directory = Path(run_directories[0])

    # Run the cave analysis https://automl.github.io/CAVE/stable
    subprocess.Popen(
        [
            "cave",
            run_directory,
            "--ta_exec_dir",
            Path().resolve(),
            "--output",
            cave_output_path,
            "--pimp_interactive",
            "off",
            "--skip",
            "ablation",
            "configurator_footprint",
            "parallel_coordinates",
            "cost_over_time",
            "budget_correlation",
        ]
    ).wait()


if __name__ == "__main__":
    logger = get_logger("ProSeCo CAVE")
    output_path = let_user_select_subdirectory(
        Path(__file__).parent.resolve() / "output"
    )
    cave_output_path = output_path / "cave"
    if not cave_output_path.exists():
        cave_output_path.mkdir(parents=True)
        run_cave_analysis(output_path)
        logger.info("CAVE analysis has finished.")
    else:
        logger.info(
            f"CAVE output directory already exists: {cave_output_path.relative_to(Path().resolve())}"
        )
    start_web_server(cave_output_path)
