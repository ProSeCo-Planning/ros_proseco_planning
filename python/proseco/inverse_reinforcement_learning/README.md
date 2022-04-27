# ProSeCo Inverse Reinforcement Learning

The ProSeCo Inverse Reinforcement Learning is a tool tailored for learning the reward function for the of the ProSeCo planner by expert trajectories.

## Setup
Since it heavily relies on data generation by the ProSeCo planner it is required to conduct the setup steps for the [evaluator](../evaluator/README.md#Setup).

## General program structure
The general structure of the program can be described as follows. Using the evaluator the IRL procedure solves the forward RL problem.

This project uses the evaluator Head-class in [evaluator](../evaluator) to spin up all MCTS nodes using the `ray` framework and finally solve the respective forward RL problem.

## Get started Training

The IRL training can then be started from the `python/inverse_reinforcement_learning` folder by 
```bash
python main.py
```
The IRL uses the `python/proseco/evaluator/evaluator.py` under-the-hood to solve the forward RL problem. The evaluator config file [evaluator/config.json](../../../config/evaluator/config.json) is used as a base config (mainly for specifying the ray_cluster, as well as the underlying options file). Other values (number runs, scenarios, etc.) are overwritten by `trainingEngine.py`. Except for the `"ray_cluster"`, **all relevant parameters** for the IRL training are specified in [inverse_reinforcement_learning/config.json](../../../config/inverse_reinforcement_learning/config.json). See the section below for a detailed description of every setting.

Before the IRL training can begin, you first need to generate expert trajectories. This project loads all expert trajectories (stored as a pickle file) from `/tmp/${USER}/irl/experts/pickle_files`. A single trajectory contains the speed, velocity, heading and more for every step and every agent of a given scenario. Additionally, for visualization purposes, a more detailed annotated trajectory (basically the `trajectory_annotated.json`) is stored inside the pickle files as well. The simplest way to create new experts is to change the `create_experts` setting to `true` and run [main.py](main.py). Not ideal experts can be filtered out manually by launching the dashboard under `python/proseco/dashboard`, selecting the `Eval_..._irl_experts` run and sorting all entries by the output path name. The run-id in the output path name corresponds to the id of the generated pickle file inside `/tmp/${USER}/irl/experts/pickle_files`. For example, `/tmp/${USER}/proseco_evaluator_output/Eval_..._irl_experts/result_inum_1/result_inum_1_0019of0048_on_48` corresponds to the trajectory file `/tmp/${USER}/irl/experts/pickle_files/SC01/expert_message_19.p`. If this trajectory should not be used for the IRL training, this pickle file can be simply deleted from its directory. To verify that only "good" trajectories are used for the IRL training, the following script can be used: `python utils/evaluator.py`

The generated evaluator output from `/tmp/${USER}/proseco_evaluator_output/` is solely used for visualization purposes (i.e. the dashboard). The IRL training does not require any file from this directory.

## Get started Inference
The inference is done to test a learned reward and create optimal trajectories with the MCTS Nodes for that reward.
The inference can then be started from the `python/proseco/inverse_reinforcement_learning` folder by running: 
```bash
python inferenceEngine.py
```
As with the training some variables need to be set correctly for proper working (see inferenceEngine.py variables below).

## Main Classes/Python files

- `trainingEngine.py`: Implements the `TrainingEngine` class which inherits the evaluator Head class and is responsible for the communication with the MCTS Nodes. Furthermore the `TrainingEngine` class has an instance of a child of the `BaseIRLModel` model class for example `LinearIRL`. Besides the communication it also calls the `Evaluator` and the `VideoCreator` to make plots/evaluations and videos for the current run.

- `Linear IRL`:
    - `irl_models/linearIrl.py`: Implements the `LinearIRL` class (implements the interface `BaseIRLModel`) and holds the tensorflow model for the linear reward. It is the method which is based on guided cost learning (Finn et al. 2016). It has two main components which are variable which is the reward features and the sampling policy. These two have seperate classes and are set inside the `LinearIRL` class with `self.reward_model` and `self.sampling_policy`. This class is responsible for performing a paramater update on the reward with the main method `self.make_update(trajectory_messages)` which gets a list of trajectory messages from the `TrainingEngine` and given back the new reward parameters after the update. Besides, this class also makes all the tensorboard logging.

    - `reward_models/linear_rewards/linearIrlReward.py`: Implements the `LinearIrlReward` which implements the interface `BaseLinearReward` and is given to a linear `BaseIrlModel` object as `self.reward_model` variable. It implements all methods that are dependend on the concrete features of the linear reward for example extracting the feature values from the trajectory message with the method `self.features_to_vec`. This reward only considers egocentric features.

    - `reward_models/linear_rewards/linearIrlRewardCooperative.py`: Implements the `LinearIrlRewardCooperative` which implements the interface `BaseLinearReward` and is given to a linear `BaseIrlModel` object as `self.reward_model` variable. It implements all methods that are dependend on the concrete features of the linear reward for example extracting the feature values from the trajectory message with the method `self.features_to_vec`. This reward also considers features of the other agents (cooperative).

- `Nonlinear IRL`:
    - `irl_models/nonLinearIrl.py`: Implements the `NonLinearIRL` class and holds the reward neural network as tensorflow graph. As the linear reward, it holds a `self.reward_model` and a `self.sampling_policy`, where in this case the reward model specifies the input of the neural network (featurized input and size of the input).
    
    - `reward_models/non_linear_rewards/nonLinearIrlReward.py`: Implements the `NonLinearIrlReward` class which implements the `BaseNonLinearReward` interface. It is mainly responsible to extract and specfiy the featurized input to the neural net. With the `self.features_to_vec` it extracts the input features for the neural network from a trajectory_message. It therefore also determines the size of the input layer in the `NonLinearIRL` method.

- `Sampling Policies`:
    - `sampling_policies/qSamplingPolicy.py`: Implements the `QSamplingPolicy` class which implements the interface `BaseSamplingPolicy`. It specifies everything that is depended on the sampling procedure of the proposal distribution. Concretly it holds the method `self.calculate_trajectory_likelihood` to get the likelihood q(tau) for a given trajectory. Furthermore it holds the variable `self.finalSelectionPolicy` which specifies the final selection policy inside the MCTS. An instance of this class or another child class of `BaseSamplingPolicy` is used in the nonlinear IRL as well as in the linear IRL model as `self.sampling_policy`.

## Configuring the IRL
The config that is used in the IRL can be found here [config.json](../../../config/inverse_reinforcement_learning/config.json), the .cue file includes default values and comments [config.cue](../../../config/inverse_reinforcement_learning/config.cue).
### Variables settable in inferenceEngine.py

- `linear_model`: boolean if linear reward should be tested or nonlinear

- `parameter_path`: path to subdir where the parameters lie - for a linear model parameters.txt is searched in this dir and for the nonlinear reward w1.txt, w2.txt and b1.txt
   
- `compare_with_experts`: flag if sampled optimal trajectories should be compared to expert trajectories (if some exist for the same scenario) -> if true expert paths must be set correcly

- `expert_pickle_folder`: path to subdir where the expert pickle files lay (only needs to be set if compare with experts is true) 

- `scenario_name`: scenario name -> should be set equal as given in scripts/services.config

- `output_path`: output folder where the metric summaries and plots should be put

- `inference_name`: name of the inference run

- `number_of_samples`: number of sampled optimal trajectories - should be multiple of number of slaves

        
