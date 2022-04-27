package p_actionSpace

#ActionSpace: (#ActionSpaceRectangle)

// types of action spaces (the type for an action space must also be specified in the #ActionSpace... definition)
#ActionSpace: type: string | ("rectangle")

#ActionSpaceRectangle: {
	type:                "rectangle"
	max_velocity_change: float & >=0.0 | *5.0       // maximum longitudinal velocity change
	max_lateral_change:  float & >=0.0 | *5.0       // maximum lateral position change
	delta_velocity:      float & >=0.0 | *(5.0 / 3) // velocity change that specifies the bound between the action class "do_nothing" and "accelerate"
}
