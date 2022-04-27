
#include <ros/console.h>
#include <ros/time.h>
#include <cassert>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"
#include "proseco_planning/action/action.h"
#include "proseco_planning/action/actionClass.h"
#include "proseco_planning/action/noiseGenerator.h"
#include "proseco_planning/agent/agent.h"
#include "proseco_planning/agent/vehicle.h"
#include "proseco_planning/collision_checker/collisionChecker.h"
#include "proseco_planning/config/computeOptions.h"
#include "proseco_planning/config/configuration.h"
#include "proseco_planning/config/outputOptions.h"
#include "proseco_planning/config/scenarioOptions.h"
#include "proseco_planning/exporters/exporter.h"
#include "proseco_planning/math/mathlib.h"
#include "proseco_planning/monteCarloTreeSearch.h"
#include "proseco_planning/node.h"
#include "proseco_planning/scenarioEvaluation.h"
#include "proseco_planning/trajectory/trajectorygenerator.h"
#include "proseco_planning/util/alias.h"
#include "proseco_planning/util/utilities.h"
#include "ros_proseco_planning/prosecoPlanner.h"

namespace proseco_planning {

/**
 * @brief Constructs a new ProSeCo Planner object.
 *
 * @param packagePath The path to the ros_proseco_planning package.
 * @param optionsArg The path of the options file relative to
 * ros_proseco_planning/config/options.
 * @param scenarioArg The path of the scenario file relative to
 * ros_proseco_planning/config/scenarios.
 * @param absolute_path A flag that indicates whether or not to use absolute paths instead of
 * relative to the ros_proseco_planning package.
 */
ProSeCoPlanner::ProSeCoPlanner(const std::string& packagePath, const std::string& optionsArg,
                               const std::string& scenarioArg, const bool absolute_path)
    : packagePath(packagePath), optionsArg(optionsArg), scenarioArg(scenarioArg) {
  std::string optionsPath{"/config/options/"};
  std::string scenarioPath{"/config/scenarios/"};

  // Load options
  const auto& options =
      config::Options::fromJSON(load_config(optionsArg, optionsPath, packagePath, absolute_path));
  ROS_INFO("Options initialized");
  // Set the seed for the thread-safe random engine to generate randomness (needs to be loaded
  // before the scenarios, since the engine is used by them)
  math::Random::setRandomSeed(options.compute_options.random_seed);

  // Load scenario
  const auto& scenario = config::Scenario::fromJSON(
      load_config(scenarioArg, scenarioPath, packagePath, absolute_path));
  ROS_INFO("Scenario initialized");

  // Assign the config to the planner
  m_cfg = Config::create(scenario, options);

  // Initialize the correct exporter class depending on the data format
  if (oOpt().export_format != config::exportFormat::NONE)
    m_exporter = Exporter::createExporter(oOpt().output_path, oOpt().export_format);

  // Initialize the noise generator
  m_noiseGenerator = std::make_unique<NoiseGenerator>();

  // Initialize the environment
  m_environment = std::make_unique<Node>(sOpt().agents);
  // Check that starting positions do not produce invalid states or collisions
  const auto& [notInvalid, notColliding] = m_environment->validateInitialization();
  if (!notInvalid) {
    throw std::runtime_error("Initial agent positions are invalid. Aborting");
  } else if (!notColliding) {
    throw std::runtime_error("Initial agent positions produce collisions. Aborting.");
  }

  ROS_INFO("Successfully initialized proseco planner");
}

/**
 * @brief Loads a config from a JSON string.
 * @details It either loads it directly from the commandline, from an absolute path or from a path
 * relative to the ros_proseco_planning package.
 *
 * @param config The config to be loaded, either a .json file or a JSON string.
 * @param configPath The path of the config file.
 * @param packagePath The ros_proseco_planning package path.
 * @param absolute_path A flag that indicates whether or not to use absolute paths instead of
 * relative to the ros_proseco_planning package.
 * @return json The loaded JSON config.
 */
json ProSeCoPlanner::load_config(const std::string& config, const std::string& configPath,
                                 const std::string& packagePath, const bool absolute_path) {
  if (!util::hasEnding(config, ".json")) {
    ROS_INFO_STREAM("CONFIG IS NOT A FILE");
    return json::parse(config);
  } else if (absolute_path) {
    ROS_INFO_STREAM("Config: " + config);
    return util::loadJSON(config);
  } else {
    ROS_INFO_STREAM("Config: " + packagePath + configPath);
    return util::loadJSON(packagePath + configPath + config);
  }
}

/**
 * @brief Determines the agents of interest (i.e. the agents that are within sensor range) for a
 * specific agent.
 *
 * @param ego_agent The agent for which to determine the agents of interest.
 * @return std::vector<Agent> The agents that are within the region of interest.
 */
std::vector<Agent> ProSeCoPlanner::agents_of_interest(const Agent& ego_agent) {
  std::vector<Agent> agents;
  agents.reserve(m_environment->m_agents.size());
  for (auto& agent : m_environment->m_agents) {
    // if it is not the agent itself
    if (ego_agent.m_id != agent.m_id) {
      agent.is_ego = false;
      if (std::abs(agent.m_vehicle.m_positionX - ego_agent.m_vehicle.m_positionX) <
          cOpt().region_of_interest) {
        agents.emplace_back(agent);
      }
    }
    // the ego agent is always in the ROI
    else {
      agents.emplace_back(ego_agent);
    }
  }
  return agents;
}

/**
 * @brief Evaluates whether the planner has reached a terminal state.
 *
 * @return true If the environment reached a collision or invalid state, as well as when the maximum
 * number of steps or maximum duration has been reached. But also, when the desire is fulfilled or
 * the terminal condition of the scenario is met.
 * @return false Otherwise.
 */
bool ProSeCoPlanner::isTerminal() const {
  // terminate if a collision has occurred or an invalid state has been reached
  if (m_environment->m_collision || m_environment->m_invalid || max_steps_reached() ||
      max_duration_reached()) {
    return true;
  }
  // terminate if desire is fulfilled
  else if (cOpt().end_condition == "desire") {
    return m_environment->m_terminal && isScenarioTerminal(m_environment.get());
    // terminate if terminal conditions specfied in scenario are met
  } else if (cOpt().end_condition == "scenario") {
    return isScenarioTerminal(m_environment.get());
  } else {
    return false;
  }
}

/**
 * @brief Checks whether the maximum number of planning steps for a scenario have been reached.
 *
 * @return true If the maximum has been reached.
 * @return false Otherwise.
 */
bool ProSeCoPlanner::max_steps_reached() const {
  return cOpt().max_scenario_steps != 0 && m_step >= cOpt().max_scenario_steps;
}

/**
 * @brief Checks whether the maximum planning duration for a scenario has been reached.
 *
 * @return true If the maximum has been reached.
 * @return false Otherwise
 */
bool ProSeCoPlanner::max_duration_reached() const {
  return cOpt().max_scenario_duration != 0 && m_duration.toSec() > cOpt().max_scenario_duration;
}

/**
 * @brief Saves the result and the config of the planner to the disk.
 *
 */
void ProSeCoPlanner::save() const {
  save_config();
  if (oOpt().hasExportType("result")) save_result();
}

/**
 * @brief Saves the options and scenario config to the disk.
 * @details This is needed for the analysis and to reproduce a specific run.
 *
 */
void ProSeCoPlanner::save_config() const {
  ROS_INFO_STREAM("Writing output to: " + oOpt().output_path);

  util::saveJSON(oOpt().output_path + "/options_output", m_cfg->options.toJSON());
  util::saveJSON(oOpt().output_path + "/scenario_output", sOpt().toJSON());
}

/**
 * @brief Saves the result to the disk.
 *
 */
void ProSeCoPlanner::save_result() const {
  json jResult;

  jResult["scenario"]          = sOpt().name;
  jResult["carsCollided"]      = m_environment->m_collision;
  jResult["carsInvalid"]       = m_environment->m_invalid;
  jResult["desiresFulfilled"]  = m_environment->m_terminal;
  jResult["maxSimTimeReached"] = max_duration_reached();
  jResult["maxStepsReached"]   = max_steps_reached();
  // final step is the last step that has been executed
  jResult["finalstep"] = m_step - 1;
  // normalize over steps to get a fully comparable reward value for one run of the MCTS
  // m_step is incremented at the end of the plan method
  jResult["normalizedEgoRewardSum"]  = m_normalizedEgoRewardSum / m_step;
  jResult["normalizedCoopRewardSum"] = m_normalizedCoopRewardSum / m_step;
  jResult["normalizedStepDuration"]  = m_duration.toSec() / m_step;

  util::saveJSON(oOpt().output_path + "/result", jResult);
}

/**
 * @brief Determines the minimum size of the action set sequence of the decentralized plans.
 *
 * @param decentralizedActionSetSequences The different action set sequences for each agent.
 * @return size_t
 * @todo refactor/doc
 */
size_t ProSeCoPlanner::getMinimumSequenceSize(
    const std::vector<ActionSetSequence>& decentralizedActionSetSequences) {
  size_t minSequenceSize = decentralizedActionSetSequences[0].size();
  for (const auto& sequences : decentralizedActionSetSequences) {
    if (sequences.size() < minSequenceSize) {
      minSequenceSize = sequences.size();
    }
  }
  return minSequenceSize;
}

/**
 * @brief Selects the action set sequence of each respective agent from the decentralized planning
 * results.
 *
 * @param decentralizedActionSetSequences
 * @param agentsROI
 * @return ActionSetSequence
 * @todo refactor/doc
 */
ActionSetSequence ProSeCoPlanner::mergeDecentralizedActionSetSequences(
    const std::vector<ActionSetSequence>& decentralizedActionSetSequences,
    std::vector<std::vector<Agent>>& agentsROI) const {
  // create action set sequence vector, where every entry represents the actions that all agents
  // took at this node.
  ActionSetSequence actionSetSequence;
  // decentralized action set sequences may have a different depth, so we determine the minimum
  // value over all returned action set sequences.
  const auto minSequenceSize = getMinimumSequenceSize(decentralizedActionSetSequences);
  for (size_t sequenceIdx{}; sequenceIdx < minSequenceSize; ++sequenceIdx) {
    ActionSet actionSet;
    // for every decentralized node, only get the returned action of the "ego agent" of the node.
    for (size_t nodeIdx{}; nodeIdx < m_environment->m_agents.size(); ++nodeIdx) {
      // rootAgents are all agents in this decentralized node.
      auto& rootAgents            = agentsROI[nodeIdx];
      auto decentralizedActionSet = decentralizedActionSetSequences[nodeIdx][sequenceIdx];

      // find the action set of the "ego" agent of the decentralized node
      for (size_t agentIdx{}; agentIdx < rootAgents.size(); ++agentIdx) {
        if (m_environment->m_agents[nodeIdx].m_id == rootAgents[agentIdx].m_id) {
          actionSet.emplace_back(decentralizedActionSet[agentIdx]);
          break;
        }
      }
    }
    actionSetSequence.emplace_back(actionSet);
  }
  return actionSetSequence;
}

/**
 * @brief Overrides the actions chosen by predefined agents.
 * @details If an agent is predefined an action will still be chosen based on the final selection
 * policy, so in order to make sure that it actually executes a predefined action it is overwritten
 * here.
 *
 * @param actionSet The action set that is to be executed.
 */
void ProSeCoPlanner::overrideActionsPredefined(ActionSet& actionSet) const {
  for (size_t agentIdx{}; agentIdx < sOpt().agents.size(); ++agentIdx) {
    // modify output
    if (sOpt().agents[agentIdx].is_predefined) {
      actionSet.at(agentIdx) =
          std::make_shared<Action>(Action(ActionClass::DO_NOTHING, 0.0f, 0.0f));
    }
  }
}

/**
 * @brief Accumulates the ego as well as cooperative rewards over the scenario.
 * @details Normalizes the sum using the number of agents in the scenario to make different
 * scenarios comparable to each other.
 *
 */
void ProSeCoPlanner::accumulate_reward() {
  float egoRewardSum =
      std::accumulate(m_environment->m_agents.begin(), m_environment->m_agents.end(), 0.0f,
                      [](float sum, const Agent& agent) { return (sum + agent.m_egoReward); });

  float coopRewardSum =
      std::accumulate(m_environment->m_agents.begin(), m_environment->m_agents.end(), 0.0f,
                      [](float sum, const Agent& agent) { return (sum + agent.m_coopReward); });
  // since the coop reward is already a weighted sum over all agents, we norm the sum with the
  // number of agents to keep the cooperation factors
  // coopRewardSum /= m_environment->m_agents.size();

  // norm with agents so the reward can be used to compare over scenarios with varying amounts
  // of agents
  egoRewardSum /= m_environment->m_agents.size();
  coopRewardSum /= m_environment->m_agents.size();

  // sum rewards over all steps; at the end we norm over the final amount of steps to be
  // able to compare the same scenario with different amount of steps
  m_normalizedEgoRewardSum += egoRewardSum;
  m_normalizedCoopRewardSum += coopRewardSum;
}

/**
 * @brief Starts the actual search of the Monte Carlo Tree Search.
 *
 */
void ProSeCoPlanner::plan() {
  // vector of actions for each agent, in each mcts
  ActionSetSequence actionSetSequence;
  std::vector<ActionSetSequence> decentralizedActionSetSequences;
  std::vector<std::vector<Agent>> agentsROI;

  // TIMERS TO TRACK PERFORMANCE
  ros::Time startTime;
  ros::Time endTime;
  ros::Duration stepDuration;

  //### START OF THE SEARCH
  while (!ProSeCoPlanner::isTerminal()) {
    // calculate the best action set using MCTS
    // Measure the time since the start of the scenario
    startTime     = ros::Time::now();
    auto rootNode = std::make_unique<Node>(m_environment.get());
    if (cOpt().region_of_interest > 0.0f) {
      decentralizedActionSetSequences.clear();
      agentsROI.clear();
      // initialize each root node for each agent with their respective ROI
      for (const auto& agent : m_environment->m_agents) {
        auto _rootNode      = std::make_unique<Node>(m_environment.get());
        _rootNode->m_agents = agents_of_interest(agent);
        agentsROI.emplace_back(_rootNode->m_agents);
        decentralizedActionSetSequences.emplace_back(
            computeActionSetSequence(std::move(_rootNode), m_step));
      }
      actionSetSequence =
          mergeDecentralizedActionSetSequences(decentralizedActionSetSequences, agentsROI);
    } else {
      actionSetSequence = computeActionSetSequence(std::move(rootNode), m_step);
    }

    endTime      = ros::Time::now();
    stepDuration = endTime - startTime;
    m_duration += stepDuration;
    ROS_INFO_STREAM("step " << m_step << " took\t" << stepDuration.toSec() << " seconds");

    ActionSet actionSet;
    if (actionSetSequence.empty()) {
      ROS_FATAL_STREAM("TOO FEW ITERATIONS. ABORTING.");
      throw std::runtime_error("Planner did not produce a valid plan.");
    } else {
      actionSet.clear();
      actionSet = actionSetSequence[0];
    }

    // add noise to actions, if enabled
    if (cOpt().action_noise.active) {
      actionSet = m_noiseGenerator->createNoisyActions(actionSet);
    }
    // modify output if other vehicles are predefined (keep velocity)
    overrideActionsPredefined(actionSet);

    // Save State of rootNode for exporting the single shot plan
    auto singleShotNode = std::make_unique<Node>(m_environment.get());

    // CREATE COLLISION CHECKER
    auto collisionChecker =
        CollisionChecker::createCollisionChecker(cOpt().collision_checker, 0.0f);

    // CREATE TRAJECTORY GENERATOR
    auto trajectoryGenerator =
        TrajectoryGenerator::createTrajectoryGenerator(cOpt().trajectory_type);

    // execute the action set
    m_environment->executeActions(actionSet, *collisionChecker, *trajectoryGenerator, true);
    // EXPORT
    if (oOpt().export_format != config::exportFormat::NONE) {
      // save the current step to the trajectory
      m_exporter->exportTrajectory(m_environment.get(), actionSet, m_step);
      // save data for inverse reinforcement learning
      m_exporter->exportIrlTrajectory(m_environment.get(), actionSet, m_step);

      // export the single shot plan
      if (oOpt().hasExportType("singleShot"))
        m_exporter->exportSingleShot(singleShotNode.get(), actionSetSequence, m_step);
    }

    // accumulate the reward metric
    accumulate_reward();

    // increment the step
    ++m_step;

  }  //### END OF THE SEARCH
  if (m_step == 0) {
    throw std::runtime_error("Planning finished before any steps have been executed.");
  }
  // write the trajectory to disk
  if (oOpt().export_format != config::exportFormat::NONE) {
    if (oOpt().hasExportType("trajectory"))
      m_exporter->writeData(m_step, ExportType::EXPORT_TRAJECTORY);
    if (oOpt().hasExportType("irlTrajectory"))
      m_exporter->writeData(m_step, ExportType::EXPORT_IRL_TRAJECTORY);
  }
};
}  // namespace proseco_planning