from proseco.inverse_reinforcement_learning.trajectory import Trajectory
from .baseSamplingPolicy import BaseSamplingPolicy
import numpy as np


class QSamplingPolicy(BaseSamplingPolicy):
    """
    Sampling Policy with continousSampleExpQ Selection Policy in MCTS - for terminal state futher action are simulated from the previous Q policies in the trajectory

    Instance variables:
        q_scale float -- c scale inside the Q softmax
        name string -- name of the sampling policy
        final_selection_policy string -- name of the finalSelectionPolicy in the MCTS
        drop_importance_weights bool -- boolean specifying if a likelihood should at all be calculated and used in the ML procedure (or just a uniform dist is assumed for the importance samples)
    """

    def __init__(self, drop_importance_weights):

        self.q_scale = 100.0
        self.name = "QSampling"
        self.finalSelectionPolicy = "sampleExpQ"
        self.drop_importance_weights = drop_importance_weights

    def get_name(self):
        """
        return name of q sampling policy

        Return:
            string -- name
        """
        return self.name

    def get_q_scale(self):
        """
        returns c scale in Q Softmax

        Return:
            float - c scale in Q Softmax
        """
        return self.q_scale

    def get_final_selection_policy(self):
        """
        returns the name (string tag) that is used in the MCTS to specifiy the final Selection policy

        Return:
            string -- name of the finalSelectionPolicy in the MCTS
        """
        return self.finalSelectionPolicy

    def create_noise_message(self):
        """
        Creates noise message that specifies if additional gaussian noise should be added to actions (in this q sampling method this is switched off)

        Return:
            msg.NoiseParam -- noise message given to MCTS node
        """
        return {
            "mean_y": [0],
            "sigma_y": [1],
            "mean_vx": [0],
            "sigma_vx": [1],
        }

    def create_expert_noise_message(self):
        """
        Creates noise message that specifies if additional gaussian noise should be added to actions if experts are created (in this q sampling method this is switched off)

        Return:
            msg.NoiseParam -- noise message given to MCTS node
        """
        return {
            "mean_y": [0],
            "sigma_y": [1],
            "mean_vx": [0],
            "sigma_vx": [1],
        }

    def do_drop_importance_weights(self):
        """
        Determines for the irl model if the importance weights calc by this class should be considered at all

        Return:
            bool -- bool if importance weights should be considered (is normally False)
        """
        return self.drop_importance_weights

    def get_artificial_importance_weight(self) -> float:
        """
        given the artificial importance weight to the irl model if drop_importance_weights is true - here in this case its only a uniform distribution

        Return:
            float -- artificial importance weight
        """
        return 1.0

    def calculate_trajectory_likelihood(
        self, trajectory_message: Trajectory, episode_length
    ):
        """
        Calculates the likelihood of the trajectory q(tau) that is used inside the MaxEnt irl procedure
        (initial distribution and likelihood of other agents are not used as they get reduced away in the fraction of the importance weights, see http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_12_irl.pdf)

        Arguments:
            trajectory_message msg.Trajectory -- Single agent trajectory for which the likelihood will be calculated
            episode_length int -- length of the episode (number of actions) -> is used for the further sampling of actions after a terminal state
        """

        likelihood = 1.0
        selection_likelihood_sum = 0.0
        average_selection_likelihood = 0.0
        average_counter = 0
        number_of_stages = 0

        #### Standard trajectory likelihood calc #########
        for step in trajectory_message.trajectory:
            number_of_stages += 1
            likelihood = likelihood * self.calculate_action_likelihood(step.action)
            assert likelihood > 0, "Likelihood of action is zero or NaN."
            selection_likelihood_sum += step.action.selectionLikelihood
            average_counter += 1
        average_selection_likelihood = selection_likelihood_sum / average_counter

        ##### Simulate additional actions such that every trajectory has the same length ##############
        # This is done by repeatedly sampling from the last policy of the agents until the end of the episode
        number_of_additional_stages = episode_length - len(
            trajectory_message.trajectory
        )
        replacement_agent_vec = trajectory_message.trajectory[-1]
        for _ in range(0, number_of_additional_stages):
            number_of_stages += 1
            likelihood = likelihood * self.simulate_action(replacement_agent_vec.action)

        assert number_of_stages == episode_length
        return likelihood, average_selection_likelihood

    def calculate_action_likelihood(self, action):
        """
        returns the selection likelihood of an action (could be calculated different in other q sampling methods for example when gaussian noise is added)

        Return:
            float -- likelihood of action
        """
        return action.selectionLikelihood

    def simulate_action(self, action):
        """
        Simulates an action from a given set of selection weights of action - no need to specify the action itself only the likelihood is needed

        Arguments:
            action msg.Action -- action message containing the selection weights to sample from

        Return:
            float -- likelihood of the chosen action
        """
        weights = action.selectionWeights
        if len(weights) == 0:
            # only used for deterministic trajectories (in the validation the maximal trajectories are used)
            return 1.0
        sum_weights = np.sum(weights)
        assert sum_weights > 0 and np.isfinite(
            sum_weights
        ), "The action weights sum is zero or NaN or infinity."
        likelihoods = weights / sum_weights
        chosen_likelihood = np.random.choice(likelihoods, size=1, p=likelihoods)
        return chosen_likelihood[0]
