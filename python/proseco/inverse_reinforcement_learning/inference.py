"""The entry point for the inference of the IRL algorithm. It loads the IRL config and creates the models, then runs the inference."""
import logging

from proseco.inverse_reinforcement_learning.modelBuilder import create_model, IrlConfig
from proseco.inverse_reinforcement_learning.inferenceEngine import InferenceEngine

from proseco.utility.io import get_absolute_path, load_data
from proseco.utility.ui import get_logger

if __name__ == "__main__":
    logger = get_logger("ProSeCo IRL", logging.INFO)
    logger.info("Starting ProSeCo IRL Inference")
    irl_config = IrlConfig(
        **load_data(
            get_absolute_path("config/inverse_reinforcement_learning/config.json")
        )
    )
    # If expert trajectories should be created, set the number of steps to 0.
    if irl_config.create_experts:
        irl_config.number_of_steps = 0

    irl_model = create_model(irl_config, inference_mode=True)
    irl_model.initialize()
    irl_model.build_model()
    inference_engine = InferenceEngine(irl_model, irl_config)
    inference_engine.infer()
