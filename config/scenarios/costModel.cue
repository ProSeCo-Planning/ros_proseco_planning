package p_costModel

#CostModel: (#CostContinuous | *#CostExponential | #CostLinear | #CostLinearCooperative | #CostNonLinear)

// names of the cost models (the name for a cost model must be specified in the #Cost... definition) 
#CostModel: name: string & ("costContinuous" | "costExponential" | "costLinear" | "costLinearCooperative" | "costNonLinear")

//---------------------
// Auxiliary Classes
//---------------------

_#Basic: {
	w_acceleration_x:        float // weight for x acceleration
	w_acceleration_y:        float // weight for y acceleration
	cost_collision:          float // weight for collisions
	cost_invalid_action:     float // weight for invalid actions (i.e. violating maximum steering angle or acceleration)
	cost_invalid_state:      float // weight for invalid states (i.e. driving off road)
	w_lane_center_deviation: float // weight for deviations from the center of a lane
	w_lane_change:           float // weight for lane changes
	w_lane_deviation:        float // weight for lane deviations (compared to the desired lane)
	w_velocity_deviation:    float // weight for velocity deviations (compared to the desired velocity)
	cost_enter_safe_range:   float // weight for @todo remove
	reward_terminal:         float // weight for terminal rewards (i.e. when a scenario is considered solved)
	...
}
_#Cooperative: {
	cost_collision_cooperative:          float
	cost_invalid_action_cooperative:     float
	cost_invalid_state_cooperative:      float
	w_acceleration_y_cooperative:        float
	w_lane_center_deviation_cooperative: float
	w_lane_deviation_cooperative:        float
	w_velocity_deviation_cooperative:    float
	...
}

_#NonLinear: {
	w1: [...float] // weights of the first layer 
	w2: [...float] // weights of the second layer
	...
}

//---------------------
// Cost Models
//---------------------

#CostContinuous: _#Basic & {
	name:                    "costContinuous"
	w_acceleration_x:        float | *0.0
	w_acceleration_y:        float | *-5.0
	cost_collision:          float | *-1000.0
	cost_invalid_action:     float | *0.0
	cost_invalid_state:      float | *-1000.0
	w_lane_center_deviation: float | *85.0
	w_lane_change:           float | *-10.0
	w_lane_deviation:        float | *100.0
	w_velocity_deviation:    float | *500.0
	cost_enter_safe_range:   float | *-10.0
	reward_terminal:         float | *0.0
}

#CostExponential: _#Basic & {
	name:                    "costExponential"
	w_acceleration_x:        float | *0.0
	w_acceleration_y:        float | *-5.0
	cost_collision:          float | *-1000.0
	cost_invalid_action:     float | *0.0
	cost_invalid_state:      float | *-1000.0
	w_lane_center_deviation: float | *85.0
	w_lane_change:           float | *-10.0
	w_lane_deviation:        float | *100.0
	w_velocity_deviation:    float | *500.0
	cost_enter_safe_range:   float | *-10.0
	reward_terminal:         float | *0.0
}

#CostLinear: _#Basic & {
	name:                    "costLinear"
	w_acceleration_x:        float | *0.0
	w_acceleration_y:        float | *1.87
	cost_collision:          float | *-1.666
	cost_invalid_action:     float | *-0.2683
	cost_invalid_state:      float | *-1.371
	w_lane_center_deviation: float | *2.174
	w_lane_change:           float | *0.0
	w_lane_deviation:        float | *2.218
	w_velocity_deviation:    float | *1.603
	cost_enter_safe_range:   float | *0.0
	reward_terminal:         float | *0.0
}

#CostLinearCooperative: _#Basic & _#Cooperative & {
	name:                                "costLinearCooperative"
	w_acceleration_x:                    float | *0.0
	w_acceleration_y:                    float | *2.282
	cost_collision:                      float | *-0.657
	cost_invalid_action:                 float | *0.093
	cost_invalid_state:                  float | *-0.972
	w_lane_center_deviation:             float | *2.957
	w_lane_change:                       float | *0.0
	w_lane_deviation:                    float | *1.202
	w_velocity_deviation:                float | *2.507
	cost_enter_safe_range:               float | *0.0
	reward_terminal:                     float | *0.0
	cost_collision_cooperative:          float | *-0.821
	cost_invalid_action_cooperative:     float | *-0.037
	cost_invalid_state_cooperative:      float | *-1.487
	w_acceleration_y_cooperative:        float | *3.144
	w_lane_center_deviation_cooperative: float | *3.21
	w_lane_deviation_cooperative:        float | *3.255
	w_velocity_deviation_cooperative:    float | *2.216
}

#CostNonLinear: _#NonLinear & {
	name: "costNonLinear"
	w1:   [...float] | *[
		-0.17806120216846466,
		0.30812060832977295,
		0.625923752784729,
		-0.06229365989565849,
		-0.10130176693201065,
		0.06863901764154434,
		0.5691829323768616,
		1.0066018104553223,
		-0.05364695563912392,
		-0.09741925448179245,
		0.009070416912436485,
		0.6835213899612427,
		1.0056074857711792,
		-0.007731246761977673,
		0.01457601971924305,
		-0.12292811274528503,
		-0.18855229020118713,
		-0.31741446256637573,
		-0.03791085258126259,
		0.46708041429519653,
		-0.184540256857872,
		-0.062125835567712784,
		-0.10414931923151016,
		-0.19065630435943604,
		0.2825232148170471,
		0.1785038560628891,
		-0.15176720917224884,
		-0.10173684358596802,
		-0.1477733552455902,
		0.1559510976076126,
		-0.12997466325759888,
		0.2837706208229065,
		0.9635805487632751,
		-0.051311515271663666,
		-0.08643289655447006,
		-0.14710800349712372,
		0.09559086710214615,
		0.4073842167854309,
		0.15400122106075287,
		-0.01320701651275158,
		-0.19156713783740997,
		0.6854377388954163,
		0.9407211542129517,
		-0.040553271770477295,
		0.05924730375409126,
		-0.03733581677079201,
		0.4177089333534241,
		0.7273625135421753,
		-0.08123252540826797,
		-0.045620329678058624,
	]     // weights of the first layer
	w2:   [...float] | *[
		-0.02599203772842884,
		1.2376011610031128,
		2.226767063140869,
		0.18742786347866058,
		-0.5449250936508179,
	]     // weights of the second layer
}
