package p_agent

import (
	"proseco.com:p_actionSpace"
	"proseco.com:p_costModel"
	"proseco.com:p_desire"
	"proseco.com:p_vehicle"
	"proseco.com:p_terminalCondition"
)

#Agent: {
	id:                        uint              // unique identifier
	is_predefined:             bool | *false     // flag that indicates constant behavior (i.e. constant velocity)
	cooperation_factor:        float & >=0 & <=1 // coefficient that weights the incorporation of the rewards of other agents
	action_space:              p_actionSpace.#ActionSpace
	cost_model:                p_costModel.#CostModel
	desire:                    p_desire.#Desire
	vehicle:                   p_vehicle.#Vehicle
	terminal_condition:        p_terminalCondition.#TerminalCondition
}
