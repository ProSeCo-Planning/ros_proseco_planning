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
			lane:     0
			velocity: 10.0
		}
		terminal_condition: p_terminalCondition.#TerminalCondition & {
			comparator_position_x: "larger"
			comparator_position_y: "none"
			position_x:            150.0
			position_y:            0.0
		}
		vehicle: p_vehicle.#bmw3 & {
			heading:          0.0
			position_x:       0.0
			position_y:       1.625
			velocity_x:       10.0
			velocity_y:       0.0
			sigma_position_x: 2.5
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
			position_y:            0.0
		}
		vehicle: p_vehicle.#bmw3 & {
			heading:          0.0
			position_x:       0.0
			position_y:       4.875
			velocity_x:       10.0
			velocity_y:       0.0
			sigma_position_x: 2.5
		}
	}, {
		id:                 2
		cooperation_factor: 0.5
		cost_model:         p_costModel.#CostExponential
		desire:             p_desire.#Desire & {
			lane:     2
			velocity: 10.0
		}
		terminal_condition: p_terminalCondition.#TerminalCondition & {
			comparator_position_x: "larger"
			comparator_position_y: "none"
			position_x:            150.0
			position_y:            0.0
		}
		vehicle: p_vehicle.#bmw3 & {
			heading:          0.0
			position_x:       0.0
			position_y:       8.125
			velocity_x:       10.0
			velocity_y:       0.0
			sigma_position_x: 2.5
		}
	}, {
		id:                 3
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
			position_y:            0.0
		}
		vehicle: p_vehicle.#bmw3 & {
			heading:          0.0
			position_x:       -20.0
			position_y:       4.875
			velocity_x:       10.0
			velocity_y:       0.0
			sigma_position_x: 2.5
		}
	}]
	name:      "SC14"
	obstacles: [...p_obstacle.#obs2] & [{
		id:         0
		position_x: 30.0
		position_y: 1.75
	}, {
		id:         1
		position_x: 30.0
		position_y: 8.75
	}, {
		id:         2
		position_x: 50.0
		position_y: 5.25
	}, {
		id:         3
		position_x: 70.0
		position_y: 8.75
	}, {
		id:         4
		position_x: 70.0
		position_y: 1.75
	}, {
		id:         5
		position_x: 90.0
		position_y: 5.25
	}]
	road: p_road.#threeLanes
}
