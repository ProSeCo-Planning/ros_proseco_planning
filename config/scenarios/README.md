# Scenarios

Videos of the following scenarios can be found [here](http://bugatti.fzi.de/DeCoC-MCTS-MDN/).

## Generation

Scenarios are written using [CUE](https://cuelang.org/). From these CUE definitions the required JSON scenario files can be generated by running `python generate_scenarios.py`, this requires the CUE executable, which can be downloaded [here](https://github.com/cuelang/cue/releases).

## Validation

Scenarios can be validated after creation and before use. This can be done with: `cue vet sc01.json scenario.cue`

## Longitudinal two Vehicles

### SC00

![SC00](sc00.svg)  
Avoiding a collision with static obstacles

### SC01

![SC01](sc01.svg)  
Delaying merge due to approaching vehicle in desired lane

### SC02

![SC02](sc02.svg)  
Reacting to approaching vehicle

## Merge without Obstacles

### SC03

![SC03](sc03.svg)  
Merging into moving traffic

### SC04

![SC04](sc04.svg)  
Merging into moving traffic with prior longitudinal adjustment

## Merge with Obstacles

### SC05

![SC05](sc05.svg)  
Changing lane as other vehicle needs to merge onto lane

### SC06

![SC06](sc06.svg)  
Delaying lane change as other vehicle needs to merge first

### SC07

![SC07](sc07.svg)  
Merging (1 vehicle) with 3 Vehicles

### SC08

![SC08](sc08.svg)  
Merging (1 vehicle) with 5 Vehicles

### SC09

![SC09](sc09.svg)  
Merging (2 vehicles) with 5 Vehicles

## Overtake

### SC11

![SC11](sc11.svg)  
Overtaking with oncoming traffic

## Bottleneck

### SC12

![SC12](sc12.svg)  
Passing a narrow passage with oncoming traffic

### SC13

![SC13](sc13.svg)  
Passing a narrow passage with a gap and oncoming traffic

### SC10

![SC10](sc10.svg)  
Merging into middle lane (3 lanes)

### SC16

![SC16](sc16.svg)  
Passing a narrow passage with oncoming traffic

## Obstacle Maze

### SC14

![SC14](sc14.svg)  
Obstacle Maze with 4 vehicles (3 lanes)

### SC15

![SC15](sc15.svg)  
Obstacle Maze with 8 vehicles (3 lanes)

### SC20

![SC20](sc20.svg)  
linear reward for IRL

### SC21

![SC21](sc21.svg)  
linear cooperative reward for IRL

### SC22

![SC22](sc22.svg)  
nonlinear reward for IRL
