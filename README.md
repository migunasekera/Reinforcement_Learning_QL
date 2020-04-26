# Learning optimal policies in an MDP environment
Assignment 4 for ECE 5242: Intelligent Autonomous Systems, taught by Dr. Dan Lee

## About:
The purpose of this project was to find optimal or near-optimal solutions to solve problems in complex environments.

### What is reinforcement learning?
## How to run
For acrobot, first run following:
QL.py --acrobot --optimal
followed by following to see evaluation
QL.py --acrobot

usage: QL.py [-h] [--car] [--maze] [--acrobot]

Choose the environment we are working with

optional arguments:
  -h, --help  show this help message and exit
  --car       Choose the MountainCar gym environment
  --maze      Choose the Maze environment
  --acrobot   Choose the Acrobat environment

## Results:
![Acrobot 5 seeds](./Results/acrobot_MAverage.png)

![Acrobot discounting test](./Results/acrobot_MAverage_discounting.png)

![MountainCar 5 seeds](./Results/mountaincar_MAverage.png)

![MountainCar reward shaping](./Results/mountaincar_MAverage_RShaping.png)


