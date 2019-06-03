import gym
import time
import sys
import numpy as np
import evaluation
import argparse




env = gym.make("MountainCar-v0")



LEARNING_RATE = 0.1
DISCOUNT_RATE = 0.95
EPISODES = 15000
EPSILON = 0.10
EPISODE_EVALUATION = 2000 #Evaluate the current policy at these times. Plottable


# These are the number of observations that we are subsampling our space from. This make writing our
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high) # Generates the size of observations table. 20 was randomly chosen
# Could be improved through analyzing episodes in the future
discrete_window_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE # Step sizes!
Steps = 200


q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))




def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_window_size
    return tuple(discrete_state.astype(np.int))


 # Run through episodes

 #idk why we're not waiting for it to be done)

for episode in range(EPISODES):
    done = False
    discrete_state = get_discrete_state(env.reset())
    if episode % EPISODE_EVALUATION == 0:
        # print(f"Episode {episode}")
        print("Episode:", episode)
        render = True        
    else:
        render = False
    while not done:
        values = q_table[discrete_state]
        action = evaluation.get_action_egreedy(values,EPSILON)
        # action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_RATE * max_future_q)
            q_table[discrete_state + (action, )] = new_q # This was the critical step. You are updating the current Q state, not the future one!
        elif new_state[0] >= env.goal_position: #env.goal_position = 0.5 for MountainCar-v0 environment
            q_table[new_discrete_state + (action, )] = 0 # Automatically, this is an optimal state!
            print("Done at Episode: ", episode)
        discrete_state = new_discrete_state
        



env.close()


    
        