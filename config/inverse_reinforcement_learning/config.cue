#IRLTraining: {
	create_experts:              bool | *false                                                                                                            // Whether or not expert trajectories should be created
	training_name:               string | *"linear"                                                                                                       // Name of the training and the used reward model
	scenarios:                   [...string] & [...( =~"^sc\\d\\d$")] | *["sc01", "sc02", "sc03", "sc04", "sc05", "sc06", "sc07", "sc08", "sc09", "sc10"] // List of scenarios to be used
	only_matching_scenarios:     bool | *false                                                                                                            // Whether only matching scenarios should be considered during the training. If true, the algorithm cycles through the scenarios so that each training iteration generates proposal trajectories for a specific scenario and only expert trajectories for this scenario are considered for the gradient optimization step. If false, each training iteration samples proposal trajectories for all scenarios and all expert trajectories are considered.
	number_of_q_samples:         uint | *48                                                                                                               // Number of proposal samples inside the IRL model. If create_experts is true than this is the number of experts.
	number_of_steps:             uint | *501                                                                                                              // Number of training steps
	linear_reward:               bool | *true                                                                                                             // Wether the linear IRL model should be used. If False the nonlinear IRL model will be used.
	cooperative_reward:          bool | *false                                                                                                            // Wether the cooperative IRL model should be used. If False the noncooperative IRL model will be used (only possible for linear rewards)
	learning_rate:               float | *0.005                                                                                                           // Learning rate for the gradient step
	q_scale:                     float | *100.0                                                                                                           // Q-scale in the q sampling procedure
	options_irl:                 string | *"options_irl"                                                                                                  // Name of the base options file that should be used for the IRL training
	options_experts:             string | *"options_irl_experts_2"                                                                                        // Name of the base options file that should be used for creating experts
	override_expert_cost_params: [...float] | *[1.0, 1.3, 1.9, -1.0, -1.0, -2.5, 0.3]                                                                     // Linear cost parameters for creation of experts. Using null will fall back to the parameters in the scenario config
}

#IRLTraining
