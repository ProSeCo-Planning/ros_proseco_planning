package p_options

#ComputeOptions: close({
	n_iterations:               uint | *500                                        // number of iterations the MCTS is run for
	max_search_depth:           uint | *5                                          // maximum depth the MCTS looks into the future (i.e. the planning horizon)
	max_invalid_action_samples: uint | *25                                         // maximum number of invalid actions (of a single agent) to be sampled when expanding/simulating a node.
	uct_cp:                     float | *4.0                                       // coefficient for the exploration term of UCT
	discount_factor:            float & >0 & <=1 | *0.7                            // influence of future rewards on the current action/state-value, 1 
	action_duration:            float & >0 | *2.0                                  // [s] duration of an action
	delta_t:                    float & >0 | *0.1                                  // [s] time delta between collision checks
	region_of_interest:         float & >=0 | *0.0                                 // if >0, this value acts as the maximum viewing distance and the mcts is performed for each agent individually (i.e. one tree per agent) instead of in a single (centralized) tree
	noise:                      #Noise                                             // noise that can be induced to the position of the vehicle
	action_noise:               #ActionNoise                                       // noise that can be induced to the y-position and x-velocity of the vehicle
	policy_options:             #PolicyOptions                                     // options for the policy
	parallelization_options:    #ParallelizationOptions                            // options for parallelizing the MCTS
	max_step_duration:          float & >=0 | *0.0                                 // [s] maximum duration a step is allowed to take (i.e. ensuring a certain planning frequency 0.1s => 10Hz); 0 means no maximum duration limit                                                
	max_scenario_steps:         uint | *0                                          // maximum number of steps a scenario is allowed to take (aka episode length; 0 means no maximum step limit)
	max_scenario_duration:      float & >=0 | *0.0                                 // [s] maximum duration a scenario is allowed to take (i.e. ensuring that a scenario does not run forever independent of whether a desired state is reached); 0 means no maximum duration limit   
	end_condition:              string & (*"scenario" | "infinite" | "desire")     // type of end condition 
	safety_distance:            float & >=0 | *0.0                                 // safety distance used for collision checking
	collision_checker:          string & ("circleApproximation")                   // type of collision checker
	trajectory_type:            string & (*"jerkOptimal" | "constantAcceleration") // trajectory generator specification
	random_seed:                uint | *0                                          // random seed used to generate identical results (a value of zero generates a random random seed)
})
#Noise: {
	active: bool | *false // flag that indicates whether noise is enabled
	mean:   float | *0.0  // mean of the noise
	sigma:  float | *0.15 // standard deviation of the noise
}
#ActionNoise: {
	active:   bool | *false     // flag that indicates whether noise is enabled
	mean_y:   float | *0.0      // mean of the noise
	sigma_y:  float & >0 | *0.1 // standard deviation of the noise
	mean_vx:  float | *0.0      // mean of the noise
	sigma_vx: float & >0 | *0.1 // standard deviation of the noise
}
#MoveGrouping: {
	active:                bool | *true  // flag that indicates whether move grouping is enabled
	cp:                    float | *12.0 // coefficient for the exploration term of UCT within move grouping
	final_decision:        bool | *true  // flag that indicates whether move groups are used for the final selection policy
	move_grouping_bias_pw: bool | *true  // flag that indicates whether move grouping is used to bias sampling
	move_grouping_criteria_pw: {
		active:         bool | *true            // flag that indicates whether progressive widening is based on move groups (i.e. true: action class specific values are used for criteria, false: node specific values are used for criteria)
		coefficient_pw: float & >0 | *0.55      // progressive widening coefficient
		exponent_pw:    float & >0 & <=1 | *0.4 // progressive widening exponent
	}
}
#ProgressiveWidening: {
	coefficient:  float & >0 | *0.55      // progressive widening coefficient
	exponent:     float & >0 & <=1 | *0.4 // progressive widening exponent
	max_depth_pw: uint | *2               // search depth up to which PW is being applied
}
#SearchGuide: {
	n_samples: uint | *100                         // number of samples to be drawn for the search guide calculation
	type:      string & (*"blindValue" | "random") // type of the search guide
}
#SimilarityUpdate: {
	active: bool | *false     // flag that indicates whether similarity update is enabled
	gamma:  float & >0 | *1.0 // parameter to control the size of the RBF kernel (larger values incorporate less actions, smaller values incorporate more actions)
}
#ParallelizationOptions: {
	n_threads:              uint | *1                  // number of threads the root parallelization is run with
	n_simulationThreads:    uint | *1                  // number of threads the leaf parallelization is run with
	similarity_voting:      bool | *true               // flag that indicates whether similarity voting is active for the root parallelization
	similarity_gamma:       float | *1.0               // gamma for the similarity function within the root parallelization
	simulation_aggregation: string & ("mean" | *"max") // aggregation strategy when running simulations in parallel 
}
#PolicyEnhancements: {
	progressive_widening:      #ProgressiveWidening    // settings for progressive widening
	move_grouping:             #MoveGrouping           // settings for move grouping
	search_guide:              #SearchGuide            // type of search guide for the action sampling in progressive widening
	similarity_update:         #SimilarityUpdate       // settings for similarity update
	q_scale:                   float & >0 | *100.0     // determines the sharpness of the sample-exp-q policy
	action_execution_fraction: float & >0 & <=1 | *0.4 // parameter that controls the length of a trajectory in the execution phase
}
#KernelRegressionLCB: {
	move_grouping: bool | *true // flag that indicates whether move grouping is enabled
	action: {// parameters for regression of actions
		kernel_variant: string | *"euclidean" // kernel variant
		gamma:          float & >0 | *0.2     // gamma for the similarity/kernel function (larger values decrease similarity, smaller values increase similarity)
		cp:             float | *0.5          // coefficient for the exploration term
	}
	if (move_grouping) {
		action_class: {// parameters for regression of action classes
			kernel_variant: string | *"manhattan" // kernel variant
			gamma:          float & >0 | *1.0     // gamma for the similarity/kernel function (larger values decrease similarity, smaller values increase similarity)
			cp:             float | *0.5          // coefficient for the exploration term
		}
	}
}
#PolicyOptions: {
	selection_policy:       string & ("UCTProgressiveWidening")                                                                   // Selection policy used by the MCTS
	expansion_policy:       string & ("UCT")                                                                                      // Expansion policy used by the MCTS
	simulation_Policy:      string & ("random" | *"moderate")                                                                     // Simulation policy used by the MCTS
	update_policy:          string & ("UCT")                                                                                      // Update policy used by the MCTS
	final_selection_policy: string & (*"maxActionValue" | "maxVisitCount" | "mostTrusted" | "kernelRegressionLCB" | "sampleExpQ") // Final selection policy used by the MCTS (determines how the actual action that gets executed is chosen)
	policy_enhancements:    #PolicyEnhancements                                                                                   // Policy enhancement object
	if (final_selection_policy == "kernelRegressionLCB") {
		kernel_regression_lcb: #KernelRegressionLCB
	}
}

#OutputOptions: {
	export_format: string & (*"msgpack" | "json" | "none") // data format of exported information
	export:        [...string] & [...("result" | "tree" | "childMap" | "permutationMap" | "moveGroups" | "trajectory" | "irlTrajectory" | "singleShot")] | *["result"]
	// result:      indicates whether a result .json file is exported
	// tree:        indicates whether a search tree .json file is exported
	// childmap:    indicates whether a childmap of the root node is exported
	// permutationMap: indicates whether a permutationMap of actions is exported (currently not used)
	// moveGroups:     indicates whether moveGroups are exported
	// trajectory:     indicates whether a trajectory is exported
	// irlTrajectory:  indicates whether a trajectory for inverse reinforcement learning is exported
	// singleShot:     indicates whether single shot plan is exported
	output_path: string | *"" // path of the output folder
}

// Definition of options
compute_options: #ComputeOptions
output_options:  #OutputOptions
