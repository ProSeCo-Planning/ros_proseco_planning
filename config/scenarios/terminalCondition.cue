package p_terminalCondition

#TerminalCondition: {
	position_x:            float                                               // [m] x position 
	position_y:            float | *0.0                                        // [m] y position 
	comparator_position_x: string & ("none" | "equal" | "smaller" | "larger")  // comparator defining the relation to be compared
	comparator_position_y: string & (*"none" | "equal" | "smaller" | "larger") // comparator defining the relation to be compared
}
