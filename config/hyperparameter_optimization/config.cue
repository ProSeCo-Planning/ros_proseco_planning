#Analyzer: {
	evaluation_name: string | *"" // Name of the evaluation, either an evaluation datestring or an evaluation folder
	options_plot:    bool | *true // Whether to plot the different options
	tuple_plot:      bool | *true // Whether to plot the different tuples (options and scenarios)
}

#Optimizer: {
	base_config:             string | *"example_hpot_config.json"                                                                                                                       // The base configuration file for the evluator that is used within the optimizer
	baseline_incumbent_plot: bool | *true                                                                                                                                               // Whether to plot the baseline versus the incumbent
	efficient_frontier_plot: bool | *true                                                                                                                                               // Whether to plot the efficient frontier
	best_function_value:     float | *685.0                                                                                                                                             // The best function value that is used invert the maximazation problem into a minimization problem
	hyperparameter_space:    string | *"example_hp_space.json"                                                                                                                          // The hyperparameter space that is used in the optimizer
	multicriterial_weights:  [...float] | *[1.0]                                                                                                                                        // The weights that are used to combine the different criterias of the optimization function
	number_iterations:       int | *50                                                                                                                                                  // The number of iterations that are used in the optimizer
	optimization_function:   [...string] & [...("success" | "carsCollided" | "carsInvalid" | "desiresFulfilled" | "normalizedCoopRewardSum" | "normalizedEgoRewardSum")] | *["success"] // The criterias that are used in the optimization function
	seed:                    int | *null                                                                                                                                                // The random seed that is used in the optimizer
	type:                    string & (*"smac_4HPO" | "smac_4BB" | "random" | "same_options")                                                                                           // The type of the optimizer
	validate_baseline_incumbent: {
		active:                bool | *false // Whether to validate the baseline versus the incumbent
		iterations_multiplier: int | *10     // The number of iterations that are used to validate the baseline versus the incumbent
	}
	#weights_match_feature: len(multicriterial_weights) == len(optimization_function)
	#weights_match_feature: true
}

#OptimizerComparison: {
	algorithms:        [...string] & [...("smac_4HPO" | "smac_4BB" | "random")] | *["smac_4HPO"] // The algorithms that are used in the optimizer comparison
	comparison_plot:   bool | *true                                                              // Whether to plot the comparison of the different algorithms
	iterations_budget: int | *2                                                                  // The number of iterations that are used in the optimizer comparison
	repetitions:       int | *1                                                                  // The number of repetitions that the optimizer comparison is run for
}

#Hpot: {
	name:                 string | *"example_hpopt" // Then name of the hyperparameter optimization (used e.g. for wandb)
	analyzer:             #Analyzer
	optimizer:            #Optimizer
	optimizer_comparison: #OptimizerComparison
}

#Hpot
