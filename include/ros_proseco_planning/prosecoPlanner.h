/**
 * @file prosecoPlanner.h
 * @brief The definition of the ProSeCoPlanner.
 *
 * @copyright Copyright (c) 2021
 *
 */

#pragma once

#include <ros/console_backend.h>
#include <ros/duration.h>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
using json = nlohmann::json;
#include "proseco_planning/action/noiseGenerator.h"
#include "proseco_planning/config/outputOptions.h"
#include "proseco_planning/exporters/exporter.h"
#include "proseco_planning/node.h"
#include "proseco_planning/util/alias.h"

namespace proseco_planning {
class Agent;
class Config;

/*!
 * @brief The ProSeCoPlanner class calls all necessary initializer and executes the Monte Carlo Tree
 * Search (MCTS) based planning algorithm.
 */

using config::exportFormat;

class ProSeCoPlanner {
 private:
  /// The exporter to save various data to disk.
  std::unique_ptr<Exporter> m_exporter;

  /// The path to the ROS package.
  const std::string packagePath;

  /// The file name of the options.json file.
  const std::string optionsArg;

  /// The file name of the scenario.json file.
  const std::string scenarioArg;

  /// The pointers to the root nodes of the Monte Carlo Search Tree. It is a vector since the search
  /// can be parallelized at the root node level.
  std::vector<std::unique_ptr<Node>> m_rootNodes;

  /// The pointer to the environment state for the simulation.
  std::unique_ptr<Node> m_environment;

  /// The central configuration object. It is implemented using a singleton and is thus usable
  /// throughout the program.
  const Config* m_cfg{nullptr};

  /// The counter for the current step of the scenario.
  unsigned int m_step{0};

  /// The normalized egoistic reward value over the scenario for all agents.
  float m_normalizedEgoRewardSum{0};

  /// The normalized cooperative reward value over the scenario for all agents.
  float m_normalizedCoopRewardSum{0};

  /// The duration the planner has been planning for.
  ros::Duration m_duration{0};

  /// The action noise generator. It can be used to add noise to the actions, thus making the
  /// execution of actions nondeterministic.
  std::unique_ptr<NoiseGenerator> m_noiseGenerator;

 public:
  ProSeCoPlanner(const std::string& packagePath, const std::string& optionsArg,
                 const std::string& scenarioArg, const bool absolute_path);

  static json load_config(const std::string& argument, const std::string& packagePath,
                          const std::string& argPath, const bool absolute_path);

  std::vector<Agent> agents_of_interest(const Agent& ego_agent);

  bool isTerminal() const;

  bool max_steps_reached() const;

  bool max_duration_reached() const;

  void save() const;

  void save_config() const;

  void save_result() const;

  void setLogLevel(ros::console::Level level) const;

  static size_t getMinimumSequenceSize(
      const std::vector<ActionSetSequence>& decentralizedActionSetSequences);

  ActionSetSequence mergeDecentralizedActionSetSequences(
      const std::vector<ActionSetSequence>& decentralizedActionSetSequences,
      std::vector<std::vector<Agent>>& agentsROI) const;

  void accumulate_reward();

  void overrideActionsPredefined(ActionSet& actionSet) const;

  void plan();
};
}  // namespace proseco_planning
