class BaseNonLinearReward(object):
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

    def get_expert_pickle_folder(self):
        print("ERROR - abstract method called")

    def features_to_vec(self, features, features_prev, meta_info):
        print("ERROR - abstract method called")

    def create_cost_message(self, parameters):
        print("ERROR - abstract method called")

    def create_cost_message_expert(self, parameters):
        print("ERROR - abstract method called")

    def set_T(self):
        print("ERROR - abstract method called")

    def is_cooperative(self):
        print("ERROR - abstract method called")
