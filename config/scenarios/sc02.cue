import (
	"proseco.com:p_agent"
	"proseco.com:p_costModel"
	"proseco.com:p_desire"
	"proseco.com:p_road"
	"proseco.com:p_terminalCondition"
	"proseco.com:p_vehicle"
	"proseco.com:p_scenario"
)

p_scenario.#Scenario & {
	agents: [...p_agent.#Agent] & [{
		id:                 0
		cooperation_factor: 0.5
		cost_model:         p_costModel.#CostExponential
		desire:             p_desire.#Desire & {
			lane:     1
			velocity: 12.0
		}
		terminal_condition: p_terminalCondition.#TerminalCondition & {
			comparator_position_x: "larger"
			comparator_position_y: "none"
			position_x:            150.0
			position_y:            3.5
		}
		vehicle: p_vehicle.#bmw3 & {
			position_x:       0.0
			position_y:       4.875
			velocity_x:       10.0
			velocity_y:       0.0
			heading:          0.0
			sigma_position_x: 3.7
		}
	}, {
		id:                 1
		cooperation_factor: 0.5
		cost_model:         p_costModel.#CostExponential
		desire:             p_desire.#Desire & {
			lane:     1
			velocity: 3.0
		}
		terminal_condition: p_terminalCondition.#TerminalCondition & {
			comparator_position_x: "larger"
			comparator_position_y: "none"
			position_x:            75.0
			position_y:            3.5
		}
		vehicle: p_vehicle.#bmw3 & {
			position_x:       20.0
			position_y:       4.875
			velocity_x:       3.0
			velocity_y:       0.0
			heading:          0.0
			sigma_position_x: 3.7
		}
	}]
	name: "SC02"
	obstacles: []
	road: p_road.#twoLanes
}
