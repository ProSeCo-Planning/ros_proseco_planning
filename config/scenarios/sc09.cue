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
		cooperation_factor: 0.5
		cost_model:         p_costModel.#CostExponential
		desire:             p_desire.#Desire & {
			lane:     0
			velocity: 7.0
		}
		id:                 0
		terminal_condition: p_terminalCondition.#TerminalCondition & {
			comparator_position_x: "larger"
			comparator_position_y: "none"
			position_x:            125.0
			position_y:            3.5
		}
		vehicle: p_vehicle.#bmw3 & {
			heading:          0.0
			position_x:       0.0
			position_y:       1.625
			velocity_x:       7.0
			velocity_y:       0.0
			sigma_position_x: 3.0
		}
	}, {
		cooperation_factor: 0.5
		cost_model:         p_costModel.#CostExponential
		desire:             p_desire.#Desire & {
			lane:     0
			velocity: 7.0
		}
		id:                 1
		terminal_condition: p_terminalCondition.#TerminalCondition & {
			comparator_position_x: "larger"
			comparator_position_y: "none"
			position_x:            125.0
			position_y:            3.5
		}
		vehicle: p_vehicle.#bmw3 & {
			heading:          0.0
			position_x:       30.0
			position_y:       1.625
			velocity_x:       7.0
			velocity_y:       0.0
			sigma_position_x: 1.0
		}
	}, {
		cooperation_factor: 0.5
		cost_model:         p_costModel.#CostExponential
		desire:             p_desire.#Desire & {
			lane:     0
			velocity: 7.0
		}
		id:                 2
		terminal_condition: p_terminalCondition.#TerminalCondition & {
			comparator_position_x: "larger"
			comparator_position_y: "none"
			position_x:            125.0
			position_y:            3.5
		}
		vehicle: p_vehicle.#bmw3 & {
			heading:          0.0
			position_x:       50.0
			position_y:       1.625
			velocity_x:       7.0
			velocity_y:       0.0
			sigma_position_x: 1.0
		}
	}, {
		cooperation_factor: 0.5
		cost_model:         p_costModel.#CostExponential
		desire:             p_desire.#Desire & {
			lane:     1
			velocity: 7.0
		}
		id:                 3
		terminal_condition: p_terminalCondition.#TerminalCondition & {
			comparator_position_x: "larger"
			comparator_position_y: "none"
			position_x:            125.0
			position_y:            3.5
		}
		vehicle: p_vehicle.#bmw3 & {
			heading:          0.0
			position_x:       20.0
			position_y:       4.875
			velocity_x:       10.0
			velocity_y:       0.0
			sigma_position_x: 1.0
		}
	}, {
		cooperation_factor: 0.5
		cost_model:         p_costModel.#CostExponential
		desire:             p_desire.#Desire & {
			lane:     1
			velocity: 7.0
		}
		id:                 4
		terminal_condition: p_terminalCondition.#TerminalCondition & {
			comparator_position_x: "larger"
			comparator_position_y: "none"
			position_x:            125.0
			position_y:            3.5
		}
		vehicle: p_vehicle.#bmw3 & {
			heading:          0.0
			position_x:       40.0
			position_y:       4.875
			velocity_x:       10.0
			velocity_y:       0.0
			sigma_position_x: 1.0
		}
	}]
	name:      "SC09"
	obstacles: [...p_obstacle.#obs2] & [{
		id:         0
		position_x: 60.0
		position_y: 5.25
	}, {
		id:         1
		position_x: 70.0
		position_y: 5.25
	}, {
		id:         2
		position_x: 80.0
		position_y: 5.25
	}, {
		id:         3
		position_x: 90.0
		position_y: 5.25
	}]
	road: p_road.#twoLanes
}
