package p_obstacle

import "proseco.com:p_constants"

#Obstacle: {
	id:               uint                                    // unique identifier
	position_x:       float                                   // [m] x position
	position_y:       float                                   // [m] y position
	width:            float & >0                              // [m] width
	length:           float & >0                              // [m] length
	heading:          float & >=0 & <=p_constants.PI_2 | *0.0 // [rad] heading
	random:           bool | *false                           // flag that indicates randomness in the objects attributes
	sigma_position_x: float | *0.0                            // [m] standard deviation of x position
	sigma_position_y: float | *0.0                            // [m] standard deviation of y position
	sigma_heading:    float & >=0 & <=p_constants.PI_2 | *0.0 // [rad] standard deviation of heading
	sigma_width:      float | *0.0                            // [m] standard deviation of width
	sigma_length:     float | *0.0                            // [m] standard deviation of length
}

#obs1: #Obstacle & {
	length: 10.0
	width:  3.0
}

#obs2: #Obstacle & {
	length: 4.0
	width:  2.0
}
