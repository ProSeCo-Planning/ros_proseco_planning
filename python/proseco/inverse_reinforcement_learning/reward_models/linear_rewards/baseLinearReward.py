class BaseLinearReward(object):
    def __init__(self):
        pass

    def get_name(self):
        print("ERROR - abstract method called")

    def get_feature_names(self):
        print("ERROR - abstract method called")

    def get_number_of_features(self):
        print("ERROR - abstract method called")

    def get_expert_parameters(self):
        print("ERROR - abstract method called")

    def features_to_vec(
        self, features, features_prev, meta_info, feature_vec_other_agents
    ):
        print("ERROR - abstract method called")

    def create_cost_message(self, parameters):
        print("ERROR - abstract method called")

    def create_cost_message_expert(self, parameters):
        return self.create_cost_message(parameters)

    def filter_expert_trajectories(self, expert_features):
        print("ERROR - abstract method called")

    def is_cooperative(self):
        print("ERROR - abstract method called")

    def set_T(self, T: int) -> None:
        """Sets the T parameter in the reward.

        Note
        ----------
        Often the T parameter is equal to the episode length.

        Parameters
        ----------
        T : int
            The scaling factor for the reward.
        """
        self.T = T

    def get_expert_pickle_folder(self) -> str:
        print("ERROR - abstract method called")
        return ""
