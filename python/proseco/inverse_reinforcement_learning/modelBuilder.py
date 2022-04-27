"""The model builder sets the experts as well as training paths, loads the IRL config and builds the models"""

import os
from dataclasses import dataclass
from typing import List

from proseco.inverse_reinforcement_learning.sampling_policies.qSamplingPolicy import (
    QSamplingPolicy,
)
from proseco.inverse_reinforcement_learning.reward_models.linear_rewards.linearIrlReward import (
    LinearIrlReward,
)
from proseco.inverse_reinforcement_learning.reward_models.non_linear_rewards.nonLinearIrlReward import (
    NonLinearIrlReward,
)
from proseco.inverse_reinforcement_learning.reward_models.linear_rewards.linearIrlRewardCooperative import (
    LinearIrlRewardCooperative,
)
from proseco.inverse_reinforcement_learning.irl_models.linearIrl import LinearIRLModel
from proseco.inverse_reinforcement_learning.irl_models.nonLinearIrl import (
    NonLinearIRLModel,
)
from proseco.inverse_reinforcement_learning.irl_models.utils import Optimizer

from proseco.utility.io import get_user_temp_dir


@dataclass
class IrlConfig:
    create_experts: bool  # Boolean if experts should be created
    experts_path: str  # Path to the expert trajectories
    training_name: str  # Name of the training
    scenarios: List[str]  # List of scenario names
    only_matching_scenarios: bool  # Boolean if only matching scenarios should be considered during the training. If true, the algorithm cycles through the `scenarios` so that each training iteration generates proposal trajectories for a specific scenario and only expert trajectories for this scenario are considered for the gradient optimization step. If false, each training iteration samples proposal trajectories for all `scenarios` and all expert trajectories are considered.
    number_of_q_samples: int  # Number of proposal samples inside the irl model. If create_experts is true than this is the number of experts.
    number_of_steps: int  # Number of training steps
    linear_reward: bool  # Boolean if linear irl model should be used. If False the nonlinear irl model will be used.
    cooperative_reward: bool  # Boolean if cooperative reward should be used (only possible for linear rewards)
    learning_rate: float  #  Learning rate for the training
    q_scale: float  # C scale in the q sampling procedure
    options_irl: str  # Name of the base options file that should be used for the irl training
    options_experts: str  # Name of the base options file that should be used for creating experts
    override_expert_cost_params: List[
        float
    ]  # Linear cost parameters for creation of experts. Using null will fall back to the parameters in the scenario config
    compare_with_experts: bool  # Boolean if sampled optimal trajectories should be compared to expert trajectories (if some exist for the same scenario)
    number_of_samples: int  # Number of sampled optimal trajectories
    episode_length: int  # Length of the episodes


# Set the loglevel to ERROR to suppress other TensorFlow output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def create_LinearIRLModel(irl_config: IrlConfig, work_dir: str) -> LinearIRLModel:
    """Creates a LinearIRLModel object.

    Parameters
    ----------
    irl_config : IrlConfig
        The configuration object for the IRL algorithm.
    work_dir : str
        The path to the working directory.

    Returns
    -------
    LinearIRLModel
        The LinearIRLModel object.
    """
    return LinearIRLModel(
        training_name=irl_config.training_name,
        reward_model=LinearIrlReward(
            irl_config.experts_path,
            irl_config.override_expert_cost_params,
        ),
        sampling_policy=QSamplingPolicy(drop_importance_weights=False),
        number_of_steps=irl_config.number_of_steps,
        learning_rate=irl_config.learning_rate,
        optimizer_type=Optimizer.MOMENTUM,
        work_dir=work_dir,
    )


def create_LinearCooperativeIRLModel(
    irl_config: IrlConfig, work_dir: str
) -> LinearIRLModel:
    """Creates a LinearCooperativeIRLModel object.

    Parameters
    ----------
    irl_config : IrlConfig
        The configuration object for the IRL algorithm.
    work_dir : str
        The path to the working directory.

    Returns
    -------
    LinearIRLModel
        The LinearCooperativeIRLModel object.
    """
    return LinearIRLModel(
        training_name=irl_config.training_name,
        reward_model=LinearIrlRewardCooperative(
            irl_config.experts_path,
            irl_config.override_expert_cost_params,
        ),
        sampling_policy=QSamplingPolicy(drop_importance_weights=False),
        number_of_steps=irl_config.number_of_steps,
        learning_rate=irl_config.learning_rate,
        optimizer_type=Optimizer.MOMENTUM,
        work_dir=work_dir,
    )


def create_NonLinearIRLModel(irl_config: IrlConfig, work_dir: str) -> NonLinearIRLModel:
    """Creates a NonLinearIRLModel object.

    Parameters
    ----------
    irl_config : IrlConfig
        The configuration object for the IRL algorithm.
    work_dir : str
        The path to the working directory.

    Returns
    -------
    NonLinearIRLModel
        The NonLinearIRLModel object.
    """
    return NonLinearIRLModel(
        training_name=irl_config.training_name,
        reward_model=NonLinearIrlReward(
            irl_config.experts_path,
            irl_config.override_expert_cost_params,
        ),
        sampling_policy=QSamplingPolicy(drop_importance_weights=False),
        number_of_steps=irl_config.number_of_steps,
        learning_rate=irl_config.learning_rate,
        optimizer_type=Optimizer.ADAM,
        dimension_hidden_layer=5,
        add_bias_term=False,
        use_mini_batches=False,
        mini_batch_size=32,
        initial_values_range=0.2,
        regularize_weight=0.01,
        regularize_first_hidden_layer=True,
        work_dir=work_dir,
    )


def create_model(
    irl_config: IrlConfig, inference_mode=False
) -> LinearIRLModel or NonLinearIRLModel:
    """Creates a LinearIRLModel or NonLinearIRLModel object depending on the irl_config.

    Parameters
    ----------
    irl_config : IrlConfig
        The configuration object for the IRL algorithm.
    inference_mode : bool
        Boolean if the model should be used for inference.

    Returns
    -------
    LinearIRLModel or NonLinearIRLModel
        The LinearIRLModel or NonLinearIRLModel object.
    """
    if inference_mode:
        # The path to the directory where the inference results are saved.
        work_dir = str(get_user_temp_dir() / "irl/inference")
    else:
        # The path to the directory where the training results are saved.
        work_dir = str(get_user_temp_dir() / "irl/training")
    if irl_config.linear_reward:
        if irl_config.cooperative_reward:
            return create_LinearCooperativeIRLModel(irl_config, work_dir)
        else:
            return create_LinearIRLModel(irl_config, work_dir)
    else:
        return create_NonLinearIRLModel(irl_config, work_dir)
