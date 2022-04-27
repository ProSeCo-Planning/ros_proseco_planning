package p_scenario

import (
	"proseco.com:p_road"
	"proseco.com:p_agent"
	"proseco.com:p_obstacle"
)

#Scenario: {
	name: string & =~"^SC\\d\\d$"
	road: p_road.#Road
	agents: [...p_agent.#Agent]
	obstacles: [...p_obstacle.#Obstacle]
}
