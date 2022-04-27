from enum import Enum
from typing import Dict, List, Any
from proseco.inverse_reinforcement_learning.trajectory import EvaluationRun
import numpy as np
import tensorflow.compat.v1 as tf

from proseco.inverse_reinforcement_learning.trajectory import *
from proseco.utility.ui import get_logger

logger = get_logger("ProSeCo IRL", create_handler=False)

tf.disable_v2_behavior()


class Optimizer(Enum):
    SGD = 1
    MOMENTUM = 2
    ADAM = 3


def calculate_reward(feature_vec, parameter_vec):
    """
    Calculates reward as dot product of parameters and feature count

    Arguments:
        feature_vec np.array -- feature count of a batch of trajectories
        parameter_vec {[type]} -- parameter vector

    Returns:
        np.array -- 1-dim np.array
    """
    return np.dot(feature_vec, parameter_vec)


def retrieve_feature_metrics(samples_tensor):
    """Calculates collision and invalid amount of current batch

    Arguments:
        samples_tensor np.array -- features of sampled batch

    Returns:
        np.array,np.array - tuple of 1 d np.array
    """
    percentage_collision = np.mean(samples_tensor[:, 3] > 0)
    percentage_invalid = np.mean(samples_tensor[:, 4] > 0)
    return percentage_collision, percentage_invalid


def show_feature_logs(sample_features, expert_features, feature_names):
    """
    Prints the feature counts of the current samples, the expert feature counts and the difference

    Arguments:
        sample_features np.array -- feature count of current sample
    """
    feature_count = np.mean(sample_features, axis=0)
    expert_feature_count = np.mean(expert_features, axis=0)
    difference = expert_feature_count - feature_count
    logger.debug("Features of all trajectories: ")
    logger.debug(sample_features)
    logger.debug(feature_names)
    logger.debug("Sample feature count:")
    logger.debug(feature_count)
    logger.debug("Expert feature count:")
    logger.debug(expert_feature_count)
    logger.debug("Difference:")
    logger.debug(difference)


def get_train_metrics(samples_tensor, parameter_vec, expert_features, feature_names):
    """
    Add train metrics to tensorboard

    Arguments:
        samples_tensor np.array -- feature count tensor of current samples
        parameter_vec np.array -- parameters after update step

    Returns:
        tf.summary -- tensorflow summary object which gets added to tensorboard
    """
    collision_amount, invalid_amount = retrieve_feature_metrics(samples_tensor)
    sample_rewards = calculate_reward(samples_tensor, parameter_vec)
    expert_rewards = calculate_reward(expert_features, parameter_vec)
    mean_samples_reward = np.mean(sample_rewards)
    mean_experts_reward = np.mean(expert_rewards)
    feature_count = np.mean(samples_tensor, axis=0)
    expert_feature_count = np.mean(expert_features, axis=0)
    feature_count_difference = expert_feature_count - feature_count
    num_parameters = len(parameter_vec)
    reward_summary = tf.Summary()
    reward_summary.value.add(
        tag="train/Samples_avg_reward", simple_value=mean_samples_reward
    )
    reward_summary.value.add(
        tag="train/Experts_avg_reward", simple_value=mean_experts_reward
    )
    reward_summary.value.add(
        tag="train/avg_reward_diff",
        simple_value=mean_samples_reward - mean_experts_reward,
    )
    reward_summary.value.add(
        tag="train/avg_reward_diff_percent",
        simple_value=(mean_samples_reward - mean_experts_reward) / mean_experts_reward,
    )
    reward_summary.value.add(
        tag="train/collision_amount", simple_value=collision_amount
    )
    reward_summary.value.add(tag="train/invalid_amount", simple_value=invalid_amount)
    for i in range(0, num_parameters):
        reward_summary.value.add(
            tag="train/" + feature_names[i] + "_feature_diff",
            simple_value=feature_count_difference[i],
        )
    return reward_summary


def create_generic_feature_diff_summary(
    prefix, samples_tensor, expert_features, feature_names
):
    """
    only for logging in tensorboard - creates a tensorboard summary for arbitrary feature vectors (is used for the feature vector of the sample with maximal IRL weight)
    """
    feature_count = np.mean(samples_tensor, axis=0)
    expert_feature_count = np.mean(expert_features, axis=0)
    feature_count_difference = expert_feature_count - feature_count
    num_parameters = len(feature_names)
    summary = tf.Summary()
    for i in range(0, num_parameters):
        summary.value.add(
            tag=prefix + "/" + feature_names[i] + "_feature_diff",
            simple_value=feature_count_difference[i],
        )
    return summary


def get_scenario_number_for_tag(tag):
    """
    only for logging in tensorboard - gets scenario name and retrieves the number of the scenario

    Arguments:
        tag String - string with the scenario name

    Return:
        int - number of the scenario
    """
    scenario_dict = {
        "sc1_coop": 1,
        "SC02": 2,
        "sc4_coop": 4,
        "sc7_coop": 7,
        "sc8_coop": 8,
        "SC10": 10,
    }
    if tag in scenario_dict:
        return True, scenario_dict[tag]
    else:
        return False, 0


def get_val_metrics(samples_tensor, parameter_vec, expert_features, feature_names):
    """
    Add train metrics to tensorboard

    Arguments:
        samples_tensor np.array -- feature count tensor of current samples
        parameter_vec np.array -- parameters after update step

    Returns:
        tf.summary -- tensorflow summary object which gets added to tensorboard
    """
    collision_amount, invalid_amount = retrieve_feature_metrics(samples_tensor)
    sample_rewards = calculate_reward(samples_tensor, parameter_vec)
    expert_rewards = calculate_reward(expert_features, parameter_vec)
    mean_samples_reward = np.mean(sample_rewards)
    mean_experts_reward = np.mean(expert_rewards)
    feature_count = np.mean(samples_tensor, axis=0)
    expert_feature_count = np.mean(expert_features, axis=0)
    feature_count_difference = expert_feature_count - feature_count
    num_parameters = len(parameter_vec)
    reward_summary = tf.Summary()
    reward_summary.value.add(
        tag="val/Samples_avg_reward", simple_value=mean_samples_reward
    )
    reward_summary.value.add(
        tag="val/Experts_avg_reward", simple_value=mean_experts_reward
    )
    reward_summary.value.add(
        tag="val/avg_reward_diff",
        simple_value=mean_samples_reward - mean_experts_reward,
    )
    reward_summary.value.add(
        tag="val/avg_reward_diff_percent",
        simple_value=(mean_samples_reward - mean_experts_reward) / mean_experts_reward,
    )
    reward_summary.value.add(tag="val/collision_amount", simple_value=collision_amount)
    reward_summary.value.add(tag="val/invalid_amount", simple_value=invalid_amount)
    for i in range(0, num_parameters):
        reward_summary.value.add(
            tag="val/" + feature_names[i] + "_feature_diff",
            simple_value=feature_count_difference[i],
        )
    return reward_summary


def calculate_average_initial_values_of_trajectories(
    trajectory_list: List[EvaluationRun],
):
    """Calculate statistics for start positions and velocities per agent a list of single agent trajectories

    Arguments:
        trajectory_list [ros_proseco_planning.msg.Trajectory[] -- trajectory messages

    Returns:
        dict -- summary dictionary of calculated statistics of all agents
    """
    flattened_trajectory_list = [
        agent for eval_run in trajectory_list for agent in eval_run.trajectories
    ]
    agents_ids = []
    trajectories_per_agent_dict: Dict[int, List[Trajectory]] = {}
    ## Get all agent ids
    for trajectory_message in flattened_trajectory_list:
        agent_id = trajectory_message.agentId
        if not agent_id in agents_ids:
            agents_ids.append(agent_id)
            trajectories_per_agent_dict[agent_id] = []
        trajectories_per_agent_dict[agent_id].append(trajectory_message)
    statistics_dist = {}
    for agent_id in trajectories_per_agent_dict:
        logger.info(f"Agent: {agent_id}")
        summary_dict = calculate_statistics_for_start_values(
            trajectories_per_agent_dict[agent_id]
        )
        statistics_dist["Agent: " + str(agent_id)] = summary_dict
    return statistics_dist


def calculate_statistics_for_start_values(
    single_agent_trajectory_list: List[Trajectory],
):
    """Calculate statistics for start positions and velocities for a list of single agent trajectories

    Arguments:
        single_agent_trajectory_list [ros_proseco_planning.msg.Trajectory[] -- trajectory messages

    Returns:
        dict -- summary dictionary of calculated statistics
    """
    initial_xs = []
    initial_ys = []
    initial_vxs = []
    initial_vys = []
    desired_vels = []
    for trajectory_message in single_agent_trajectory_list:
        initial_x = trajectory_message.initialState.state.posX
        initial_y = trajectory_message.initialState.state.posY
        initial_vx = trajectory_message.initialState.state.velX
        initial_vy = trajectory_message.initialState.state.velY
        desired_vel = trajectory_message.initialState.features.desired_vel
        desired_vels.append(desired_vel)
        initial_xs.append(initial_x)
        initial_ys.append(initial_y)
        initial_vxs.append(initial_vx)
        initial_vys.append(initial_vy)
    initial_xs = np.array(initial_xs)
    initial_ys = np.array(initial_ys)
    initial_vxs = np.array(initial_vxs)
    initial_vys = np.array(initial_vys)
    desired_vels = np.array(desired_vels)
    summary = {}
    summary["mean_initial_xs"] = np.mean(initial_xs)
    summary["mean_initial_ys"] = np.mean(initial_ys)
    summary["mean_initial_vxs"] = np.mean(initial_vxs)
    summary["mean_initial_vys"] = np.mean(initial_vys)
    summary["mean_desired_vels"] = np.mean(desired_vels)
    summary["std_initial_xs"] = np.std(initial_xs)
    summary["std_initial_ys"] = np.std(initial_ys)
    summary["std_initial_vxs"] = np.std(initial_vxs)
    summary["std_initial_vys"] = np.std(initial_vys)
    summary["std_desired_vels"] = np.std(desired_vels)

    logger.info(
        f"x_init      - mu:{summary['mean_initial_xs']:.2f} std: {summary['std_initial_xs']:.2f}"
    )
    logger.info(
        f"y_init      - mu:{summary['mean_initial_ys']:.2f} std: {summary['std_initial_ys']:.2f}"
    )
    logger.info(
        f"vx_init     - mu:{summary['mean_initial_vxs']:.2f} std: {summary['mean_initial_vxs']:.2f}"
    )
    logger.info(
        f"vy_init     - mu:{summary['mean_initial_vys']:.2f} std: {summary['mean_initial_vys']:.2f}"
    )
    logger.info(
        f"desVel_init - mu:{summary['mean_desired_vels']:.2f} std: {summary['std_initial_ys']:.2f}"
    )
    return summary


def check_if_collision_or_invalid_appeared_in_trajectory_message(
    run: EvaluationRun,
) -> bool:
    """Checks for a trajectory message if a collision or an invalid state appeared

    Arguments:
        EvaluationRun -- The evaluation run to check for collisions and invalid states.

    Returns:
        boolean -- true if no collision/invalid appeared
    """
    trajectory_valid = True
    for trajectory in run.trajectories:
        for agent in trajectory.trajectory:
            if agent.features.collided or agent.features.invalidState:
                trajectory_valid = False
    return trajectory_valid


def filter_trajectory_list(runs: List[EvaluationRun]) -> List[EvaluationRun]:
    """Filter out all trajectories which collisions and invalids (for expert trajectories)

    Arguments:
        trajectory_list [ros_proseco_planning.msg.Trajectory] -- trajectory messages

    Returns:
        [ros_proseco_planning.msg.Trajectory] -- filtered trajectory list
    """
    filtered_trajectories = []
    for run in runs:
        trajectory_valid = True
        for trajectory in run.trajectories:
            for agent in trajectory.trajectory:
                if agent.features.collided:
                    logger.debug("Collision detected in trajectory")
                    trajectory_valid = False
                if agent.features.invalidState:
                    logger.debug("Invalid state detected in trajectory")
                    trajectory_valid = False
                if agent.features.invalidAction:
                    logger.debug("Invalid action detected in trajectory")
                    trajectory_valid = False
        if trajectory_valid:
            filtered_trajectories.append(run)

    logger.info(f"{len(filtered_trajectories)} of {len(runs)} trajectories are valid")

    return filtered_trajectories


def calculate_average_goals_reached(runs: List[EvaluationRun], velocity_epsilon):
    """Calculates average goal reaching for complete list of trajectories

    Arguments:
        trajectory_messages [ros_proseco_planning.msg.Trajectory] -- trajectory messages
        velocity_epsilon float -- margin when to consider velocity desire reached

    Returns:
        float -- amount of single-agents trajectories where lane goal was reached
        float -- amount of single-agents trajectories where velocity goal was reached
    """
    lane_goal_reached = []
    velocity_goal_reached = []
    for run in runs:
        (
            run_lane_goal_reached,
            run_velocity_goal_reached,
        ) = check_if_goals_reached(run, velocity_epsilon)
        lane_goal_reached += run_lane_goal_reached
        velocity_goal_reached += run_velocity_goal_reached
    amount_lane_goal_reached = np.mean(np.array(lane_goal_reached, dtype=float))
    amount_velocity_goal_reached = np.mean(np.array(velocity_goal_reached, dtype=float))
    logger.debug(f"Desired lane reached:     {amount_lane_goal_reached:.2f}")
    logger.debug(f"Desired velocity reached: {amount_velocity_goal_reached:.2f}")
    return amount_lane_goal_reached, amount_velocity_goal_reached


def calculate_average_goals_reached_per_scenario(
    trajectory_messages: List[EvaluationRun], velocity_epsilon
):
    """
    calculates how often the lane and vel goal were reached

    Arguments:
        trajectory_list [ros_proseco_planning.msg.Trajectory] -- multi-agent trajectory messages
        velocity_epsilon float - margin for the velocity desire to be fulfilled

    Return:
        float - amount of all agents in all trajectories that reached their lane goal
        float - amount of all agents in all trajectories that reached their velocity goal
    """
    lane_goal_reached_per_scenario: Dict[str, list] = {}
    velocity_goal_reached_per_scenario: Dict[str, list] = {}
    for trajectory_message in trajectory_messages:
        scenario_name = trajectory_message.scenarioInfo.scenarioName
        if not (scenario_name in lane_goal_reached_per_scenario):
            lane_goal_reached_per_scenario[scenario_name] = []
            velocity_goal_reached_per_scenario[scenario_name] = []
        (
            single_message_lane_goal_reached,
            single_message_velocity_goal_reached,
        ) = check_if_goals_reached(trajectory_message, velocity_epsilon)
        lane_goal_reached_per_scenario[
            scenario_name
        ] += single_message_lane_goal_reached
        velocity_goal_reached_per_scenario[
            scenario_name
        ] += single_message_velocity_goal_reached
    amount_lane_goal_reached_per_scenario = {}
    amount_velocity_goal_reached_per_scenario = {}
    for scenario_key in lane_goal_reached_per_scenario:
        amount_lane_goal_reached_per_scenario[scenario_key] = np.mean(
            np.array(lane_goal_reached_per_scenario[scenario_key], dtype=float)
        )
        amount_velocity_goal_reached_per_scenario[scenario_key] = np.mean(
            np.array(velocity_goal_reached_per_scenario[scenario_key], dtype=float)
        )
    return (
        amount_lane_goal_reached_per_scenario,
        amount_velocity_goal_reached_per_scenario,
    )


def check_if_goals_reached(trajectory_message: EvaluationRun, velocity_epsilon):
    """Checks for all agents in trajectory if goals were reached

    Arguments:
        trajectory_message ros_proseco_planning.msg.Trajectory -- trajectory message
        velocity_epsilon float -- margin when to consider velocity desire reached

    Returns:
        list -- boolean if agents have reached lane goal
        list -- boolean if agents have reached velocity goal
    """
    number_of_agents = len(trajectory_message.trajectories)
    lane_goal_reached = []
    velocity_goal_reached = []
    last_state_action_pair = [
        agent.trajectory[-1] for agent in trajectory_message.trajectories
    ]
    for i in range(0, number_of_agents):
        abs_lane_diff = last_state_action_pair[i].features.abs_lane_diff
        abs_diff_vel_vel_des = abs(last_state_action_pair[i].features.diff_vel_vel_des)
        if abs_lane_diff == 0.0:
            lane_goal_reached.append(1)
        else:
            lane_goal_reached.append(0)
        if abs_diff_vel_vel_des < velocity_epsilon:
            velocity_goal_reached.append(1)
        else:
            velocity_goal_reached.append(0)
    return lane_goal_reached, velocity_goal_reached


def count_number_of_collisions_and_invalids(trajectory_list: List[EvaluationRun]):
    """
    count number of collisions and invalids for single agent trajectories

    Arguments:
        trajectory_list [ros_proseco_planning.msg.Trajectory] -- trajectory messages

    Return:
        float - amount of collisions appeared in all single agent trajectories
        float - amount of invalids appeared in all single agent trajectories
    """

    flattened_trajectory_list = [
        agent for eval_run in trajectory_list for agent in eval_run.trajectories
    ]
    num_collisions = 0
    num_invalids = 0
    for trajectory_message in flattened_trajectory_list:
        for agent in trajectory_message.trajectory:
            if agent.features.collided:
                num_collisions += 1
            if agent.features.invalidState:
                num_invalids += 1
    amount_collisions = float(num_collisions) / float(len(trajectory_list))
    amount_invalids = float(num_invalids) / float(len(trajectory_list))
    return amount_collisions, amount_invalids


def count_number_of_collisions_and_invalids_multi_agent(
    trajectory_list: List[EvaluationRun],
):
    """
    count number of collisions, invalids and failures for multi-agent trajectories

    Arguments:
        trajectory_list [ros_proseco_planning.msg.Trajectory] -- trajectory messages

    Return:
        float - amount of collisions appeared in all multi agent trajectories
        float - amount of invalids appeared in all multi agent trajectories
        float - amount of failures (at least one collision or invalid for one agent) appeared in all multi agent trajectories
    """
    num_collisions = 0
    num_invalids = 0
    num_failed = 0
    for trajectory_message in trajectory_list:
        has_collision = False
        has_invalid = False
        for trajectory in trajectory_message.trajectories:
            for agent in trajectory.trajectory:
                if agent.features.collided:
                    has_collision = True
                if agent.features.invalidState:
                    has_invalid = True
        if has_collision:
            num_collisions += 1
        if has_invalid:
            num_invalids += 1
        if has_collision or has_invalid:
            num_failed += 1
    amount_collisions = float(num_collisions) / float(len(trajectory_list))
    amount_invalids = float(num_invalids) / float(len(trajectory_list))
    amount_failed = float(num_failed) / float(len(trajectory_list))
    return amount_collisions, amount_invalids, amount_failed
