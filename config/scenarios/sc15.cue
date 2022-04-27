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
			position_x:            200.0
			position_y:            0.0
		}
		vehicle: p_vehicle.#bmw3 & {
			heading:          0.0
			position_x:       0.0
			position_y:       1.625
			velocity_x:       10.0
			velocity_y:       0.0
			sigma_position_x: 2.2
		}
	}, {
		id:                 1
		cooperation_factor: 0.5
		cost_model:         p_costModel.#CostExponential
		desire:             p_desire.#Desire & {
			lane:     0
			velocity: 10.0
		}
		terminal_condition: p_terminalCondition.#TerminalCondition & {
			comparator_position_x: "larger"
			comparator_position_y: "none"
			position_x:            200.0
			position_y:            0.0
		}
		vehicle: p_vehicle.#bmw3 & {
			heading:          0.0
			position_x:       20.0
			position_y:       1.625
			velocity_x:       10.0
			velocity_y:       0.0
			sigma_position_x: 2.2
		}
	}, {
		id:                 2
		cooperation_factor: 0.5
		cost_model:         p_costModel.#CostExponential
		desire:             p_desire.#Desire & {
			lane:     0
			velocity: 10.0
		}
		terminal_condition: p_terminalCondition.#TerminalCondition & {
			comparator_position_x: "larger"
			comparator_position_y: "none"
			position_x:            200.0
			position_y:            0.0
		}
		vehicle: p_vehicle.#bmw3 & {
			heading:          0.0
			position_x:       -20.0
			position_y:       1.625
			velocity_x:       10.0
			velocity_y:       0.0
			sigma_position_x: 2.2
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
			position_x:            200.0
			position_y:            0.0
		}
		vehicle: p_vehicle.#bmw3 & {
			heading:          0.0
			position_x:       0.0
			position_y:       4.875
			velocity_x:       10.0
			velocity_y:       0.0
			sigma_position_x: 2.2
		}
	}, {
		id:                 4
		cooperation_factor: 0.5
		cost_model:         p_costModel.#CostExponential
		desire:             p_desire.#Desire & {
			lane:     1
			velocity: 10.0
		}
		terminal_condition: p_terminalCondition.#TerminalCondition & {
			comparator_position_x: "larger"
			comparator_position_y: "none"
			position_x:            200.0
			position_y:            0.0
		}
		vehicle: p_vehicle.#bmw3 & {
			heading:          0.0
			position_x:       20.0
			position_y:       4.875
			velocity_x:       10.0
			velocity_y:       0.0
			sigma_position_x: 2.2
		}
	}, {
		id:                 5
		cooperation_factor: 0.5
		cost_model:         p_costModel.#CostExponential
		desire:             p_desire.#Desire & {
			lane:     1
			velocity: 10.0
		}
		terminal_condition: p_terminalCondition.#TerminalCondition & {
			comparator_position_x: "larger"
			comparator_position_y: "none"
			position_x:            200.0
			position_y:            0.0
		}
		vehicle: p_vehicle.#bmw3 & {
			heading:          0.0
			position_x:       -20.0
			position_y:       8.125
			velocity_x:       10.0
			velocity_y:       0.0
			sigma_position_x: 2.2
		}
	}, {
		id:                 6
		cooperation_factor: 0.5
		cost_model:         p_costModel.#CostExponential
		desire:             p_desire.#Desire & {
			lane:     2
			velocity: 10.0
		}
		terminal_condition: p_terminalCondition.#TerminalCondition & {
			comparator_position_x: "larger"
			comparator_position_y: "none"
			position_x:            200.0
			position_y:            0.0
		}
		vehicle: p_vehicle.#bmw3 & {
			heading:          0.0
			position_x:       0.0
			position_y:       8.125
			velocity_x:       10.0
			velocity_y:       0.0
			sigma_position_x: 2.2
		}
	}, {
		id:                 7
		cooperation_factor: 0.5
		cost_model:         p_costModel.#CostExponential
		desire:             p_desire.#Desire & {
			lane:     2
			velocity: 10.0
		}
		terminal_condition: p_terminalCondition.#TerminalCondition & {
			comparator_position_x: "larger"
			comparator_position_y: "none"
			position_x:            200.0
			position_y:            0.0
		}
		vehicle: p_vehicle.#bmw3 & {
			heading:          0.0
			position_x:       30.0
			position_y:       8.125
			velocity_x:       10.0
			velocity_y:       0.0
			sigma_position_x: 0.5
		}
	}]
	name:      "SC15"
	obstacles: [...p_obstacle.#obs2] & [{
		id:         0
		position_x: 60.0
		position_y: 4.875
	}, {
		id:         1
		position_x: 80.0
		position_y: 1.625
	}, {
		id:         2
		position_x: 80.0
		position_y: 8.125
	}, {
		id:         3
		length:     4.0
		position_x: 100.0
		position_y: 4.875
	}, {
		id:         4
		position_x: 110.0
		position_y: 4.875
	}, {
		id:         5
		position_x: 130.0
		position_y: 1.625
	}, {
		id:         6
		position_x: 130.0
		position_y: 8.125
	}, {
		id:         7
		position_x: 150.0
		position_y: 1.625
	}, {
		id:         8
		position_x: 150.0
		position_y: 4.875
	}]
	road: p_road.#threeLanes
}
