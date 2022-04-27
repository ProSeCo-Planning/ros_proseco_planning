package p_desire

#Desire: {
	velocity:             float              // [m/s] desired velocity 
	lane:                 uint & >=0         // desired lane
	lane_center_tolerance: float & >=0 | *1.0 // [m] tolerance for lane center deviation [m], so that the desire remains fulfilled
	velocity_tolerance:   float & >=0 | *2.0 // [m/s] tolerance for velocity deviation, so that the desire remains fulfilled (default value 2.0 = 7.2 / 3.6)
}
