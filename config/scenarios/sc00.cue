import (
	"proseco.com:p_agent"
	"proseco.com:p_costModel"
	"proseco.com:p_desire"
	"proseco.com:p_road"
	"proseco.com:p_terminalCondition"
	"proseco.com:p_vehicle"
	"proseco.com:p_obstacle"
	"proseco.com:p_scenario"
)

p_scenario.#Scenario & {
	agents: [...p_agent.#Agent] & [
		{
			id:                 0
			cooperation_factor: 0.0
			vehicle:            p_vehicle.#bmw3 & {
				position_x: 0.0
				position_y: 4.875
				velocity_x: 12.0
				velocity_y: 0.0
				heading:    0.0
			}
			cost_model: p_costModel.#CostExponential
			desire:     p_desire.#Desire & {
				lane:     1
				velocity: 12.0
			}
			terminal_condition: p_terminalCondition.#TerminalCondition & {
				comparator_position_x: "larger"
				comparator_position_y: "none"
				position_x:            110.0
				position_y:            0.0
			}
		},
	]
	name:      "SC00"
	obstacles: [...p_obstacle.#obs2] & [
			{
			id:         0
			position_x: 100.0
			position_y: 1.0
		},
		{
			id:         1
			position_x: 100.0
			position_y: 3.0
		},
		{
			id:         2
			position_x: 100.0
			position_y: 5.0
		},
	]
	road: p_road.#twoLanes
}
