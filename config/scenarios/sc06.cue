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
	agents: [...p_agent.#Agent] & [{
		id:                 0
		cooperation_factor: 0.5
		cost_model:         p_costModel.#CostExponential
		desire:             p_desire.#Desire & {
			lane:     1
			velocity: 10.0
		}
		terminal_condition: p_terminalCondition.#TerminalCondition & {
			comparator_position_x: "larger"
			comparator_position_y: "none"
			position_x:            150.0
			position_y:            7.0
		}
		vehicle: p_vehicle.#bmw3 & {
			heading:          0.0
			position_x:       32.0
			position_y:       1.75
			velocity_x:       10.0
			velocity_y:       0.0
			sigma_position_x: 4.0
		}
	}, {
		id:                 1
		cooperation_factor: 0.5
		cost_model:         p_costModel.#CostExponential
		desire:             p_desire.#Desire & {
			lane:     1
			velocity: 10.0
		}
		terminal_condition: p_terminalCondition.#TerminalCondition & {
			comparator_position_x: "larger"
			comparator_position_y: "none"
			position_x:            150.0
			position_y:            7.0
		}
		vehicle: p_vehicle.#bmw3 & {
			heading:          0.0
			position_x:       30.0
			position_y:       8.75
			velocity_x:       10.0
			velocity_y:       0.0
			sigma_position_x: 4.0
		}
	}]
	name:      "SC06"
	obstacles: [...p_obstacle.#obs1] & [{
		id:         0
		position_x: 60.0
		position_y: 1.75
	}, {
		id:         1
		position_x: 75.0
		position_y: 1.75
	}, {
		id:         2
		position_x: 90.0
		position_y: 1.75
	}, {
		id:         3
		position_x: 105.0
		position_y: 1.75
	}, {
		id:         4
		position_x: 120.0
		position_y: 1.75
	}, {
		id:         5
		position_x: 135.0
		position_y: 1.75
	}, {
		id:         6
		position_x: 150.0
		position_y: 1.75
	}]
	road: p_road.#threeLanes
}
