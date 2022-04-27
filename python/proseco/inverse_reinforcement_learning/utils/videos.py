import json
from proseco.utility.ui import let_user_select_subdirectory

TRAINING_PATH = "/tmp/co1539/irl/training"


def create_videos():
    """Create videos from the training data."""
    experiment_directory = let_user_select_subdirectory(TRAINING_PATH)
    trajectories_directory = experiment_directory / "trajectories"
    config = json.load(open(experiment_directory / "config.json"))
    for scenario in config["scenarios"]:
        import ffmpeg

        (
            ffmpeg.input(
                f"{trajectories_directory}/*/plot_{scenario}.png",
                pattern_type="glob",
                framerate=25,
            )
            .output(str(experiment_directory / f"{scenario}.mp4"))
            .run(overwrite_output=True)
        )
    print(
        f"Video generation complete. The videos are in the experiment directory: {experiment_directory}"
    )


if __name__ == "__main__":
    create_videos()
