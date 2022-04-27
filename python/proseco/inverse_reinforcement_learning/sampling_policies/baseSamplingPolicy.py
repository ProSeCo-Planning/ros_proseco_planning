class BaseSamplingPolicy(object):
    def __init__(self):
        pass

    def get_name(self):
        print("ERROR - abstract method called")

    def get_final_selection_policy(self):
        print("ERROR - abstract method called")

    def get_q_scale(self):
        print("ERROR - abstract method called")

    def create_noise_message(self):
        print("ERROR - abstract method called")

    def create_expert_noise_message(self):
        print("ERROR - abstract method called")

    def do_drop_importance_weights(self):
        print("ERROR - abstract method called")

    def get_artificial_importance_weight(self):
        print("ERROR - abstract method called")

    def calculate_trajectory_likelihood(self, trajectory_message, episode_length):
        print("ERROR - abstract method called")

    def calculate_action_likelihood(self, action_message):
        print("ERROR - abstract method called")

    def get_artificial_importance_weight(self) -> float:
        print("ERROR - abstract method called")
