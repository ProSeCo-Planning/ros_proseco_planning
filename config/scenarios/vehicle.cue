package p_vehicle

import "proseco.com:p_constants"

#Vehicle: {
	//name:               string
	position_x:         float                                    // [m] x position
	position_y:         float                                    // [m] y position
	velocity_x:         float                                    // [m/s] x velocity
	velocity_y:         float                                    // [m/s] y velocity
	width:              float & >0                               // [m] width
	length:             float & >0                               // [m] length
	heading:            float & >=0 & <=p_constants.PI_2 | *0.0  // [rad] heading
	random:             bool | *true                             // flag that indicates randomness in the objects attributes
	sigma_position_x:   float | *0.0                             // [m] standard deviation of x position
	sigma_position_y:   float | *0.2                             // [m] standard deviation of y position
	sigma_heading:      float & >=0 & <=p_constants.PI_2 | *0.0  // [rad] standard deviation of heading
	sigma_width:        float | *0.0                             // [m] standard deviation of width
	sigma_length:       float | *0.0                             // [m] standard deviation of length
	sigma_velocity_x:   float | *0.0                             // [m/s] standard deviation of x velocity
	sigma_velocity_y:   float | *0.0                             // [m/s] standard deviation of y velocity
	wheel_base:         float & >0                               // [m] standard deviation of wheelbase
	max_steering_angle: float & >0                               // [rad] maximum steering angle
	max_speed:          float & <=36.0 | *36.0                   // [m/s] maximum velocity (36 is equivalent to 130 km/h)
	max_acceleration:   float & <=p_constants.g | *p_constants.g // [m/s^2] maximum acceleration
}

#bmw1: #Vehicle & {
	//name:               "BMW 1 Series"
	width:              1.765
	length:             4.329
	wheel_base:         2.690
	max_steering_angle: 0.260
	max_speed:          36.0
	max_acceleration:   p_constants.g
}

#bmw3: #Vehicle & {
	//name:               "BMW 3 Series"
	width:              1.827
	length:             4.709
	wheel_base:         2.851
	max_steering_angle: 0.263
	max_speed:          36.0
	max_acceleration:   p_constants.g
}

#bmw5: #Vehicle & {
	//name:               "BMW 5 Series"
	width:              1.868
	length:             4.943
	wheel_base:         2.851
	max_steering_angle: 0.258
	max_speed:          36.0
	max_acceleration:   p_constants.g
}

#bmw7: #Vehicle & {
	//name:               "BMW 7 Series"
	width:              1.902
	length:             5.098
	wheel_base:         3.070
	max_steering_angle: 0.261
	max_speed:          36.0
	max_acceleration:   p_constants.g
}
