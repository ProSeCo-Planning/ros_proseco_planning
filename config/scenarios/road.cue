package p_road

#Road: {
	random:           bool                 // flag that indicates randomness in the objects attributes
	number_lanes:     uint & >0            // number of lanes the road consists of
	lane_width:       float & >=0 & <=10.0 // [m] lane width of each lane
	sigma_lane_width: float & >=0 & <=10.0 // [m] standard deviation of the lane width 
}

#twoLanes: #Road & {
	lane_width:       3.25
	number_lanes:     2
	random:           bool | *false
	sigma_lane_width: 0.25
}

#threeLanes: #Road & {
	lane_width:       3.5
	number_lanes:     3
	random:           bool | *false
	sigma_lane_width: 0.25
}
