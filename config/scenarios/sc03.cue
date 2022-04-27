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
			lane:     0
			velocity: 8.0
		}
		terminal_condition: p_terminalCondition.#TerminalCondition & {
			comparator_position_x: "larger"
			comparator_position_y: "none"
			position_x:            150.0
			position_y:            3.5
		}
		vehicle: p_vehicle.#bmw3 & {
			position_x:       60.0
			position_y:       1.625
			velocity_x:       8.0
			velocity_y:       0.0
			heading:          0.0
			sigma_position_x: 2.3
		}
	}, {
		id:                 1
		cooperation_factor: 0.5
		cost_model:         p_costModel.#CostExponential
		desire:             p_desire.#Desire & {
			lane:     0
			velocity: 8.0
		}
		terminal_condition: p_terminalCondition.#TerminalCondition & {
			comparator_position_x: "larger"
			comparator_position_y: "none"
			position_x:            150.0
			position_y:            3.5
		}
		vehicle: p_vehicle.#bmw3 & {

			position_x:       45.0
			position_y:       4.875
			velocity_x:       8.0
			velocity_y:       0.0
			heading:          0.0
			sigma_position_x: 2.3
		}
	}, {
		id:                 2
		cooperation_factor: 0.5
		cost_model:         p_costModel.#CostExponential
		desire:             p_desire.#Desire & {
			lane:     0
			velocity: 8.0
		}
		terminal_condition: p_terminalCondition.#TerminalCondition & {
			comparator_position_x: "larger"
			comparator_position_y: "none"
			position_x:            150.0
			position_y:            3.5
		}
		vehicle: p_vehicle.#bmw3 & {
			position_x:       30.0
			position_y:       1.625
			velocity_x:       8.0
			velocity_y:       0.0
			heading:          0.0
			sigma_position_x: 2.3
		}
	}]
	name: "SC03"
	obstacles: []
	road: p_road.#twoLanes
}
