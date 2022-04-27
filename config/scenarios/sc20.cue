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
		cooperation_factor: 0.0
		cost_model:         p_costModel.#CostLinear
		desire:             p_desire.#Desire & {
			lane:                  1
			velocity:              12.0
			lane_center_tolerance: 0.5
			velocity_tolerance:    1.0
		}
		terminal_condition: p_terminalCondition.#TerminalCondition & {
			comparator_position_x: "larger"
			comparator_position_y: "none"
			position_x:            100.0
			position_y:            0.0
		}
		vehicle: p_vehicle.#bmw3 & {
			heading:    0.0
			position_x: 0.0
			position_y: 4.875
			velocity_x: 12.0
			velocity_y: 0.0
		}
	}, {
		id:                 1
		cooperation_factor: 0.0
		cost_model:         p_costModel.#CostLinear
		desire:             p_desire.#Desire & {
			lane:                  1
			velocity:              8.0
			lane_center_tolerance: 0.5
			velocity_tolerance:    1.0
		}
		terminal_condition: p_terminalCondition.#TerminalCondition & {
			comparator_position_x: "larger"
			comparator_position_y: "none"
			position_x:            100.0
			position_y:            0.0
		}
		vehicle: p_vehicle.#bmw3 & {
			heading:    0.0
			position_x: 10.0
			position_y: 1.625
			velocity_x: 5.0
			velocity_y: 0.0
		}
	}]
	name: "SC20"
	road: p_road.#twoLanes & {
		random: true
	}
}
