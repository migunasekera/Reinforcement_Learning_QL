import gym
import time
import sys
import numpy as np
import evaluation
import argparse
from maze import *







def get_discrete_state(env, state, DISCRETE_OS_SIZE):
    
    discrete_window_size = (env.observation_space.high - env.observation_space.low)  / DISCRETE_OS_SIZE # Step sizes!
    # fudge_factor = (0.01) * env.observation_space.low # This doesn't work b/c the number could be negative
    discrete_state = (state - env.observation_space.low) / discrete_window_size
    # print(discrete_state)
    return tuple(discrete_state.astype(np.int))


def train_gym(env,EPISODES):
    '''
    Input: Env: Gym environment

    This will train a vlaue 
    '''

    LEARNING_RATE = 0.1
    DISCOUNT_RATE = 0.95
    EPSILON = 0.10
    EPISODE_EVALUATION = 500 #Evaluate the current policy at these times. Plottable
    DISCRETE_OS_SIZE = [20] * len(env.observation_space.high) # Generates the size of observations table. 20 was randomly chosen
    
    

    q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))
    # high = get_discrete_state(env, env.observation_space.high, DISCRETE_OS_SIZE)
    # print(get_discrete_state(env, env.observation_space.low, DISCRETE_OS_SIZE))
    # print(q_table[high])
    # sys.exit()

    for episode in range(EPISODES):
        done = False
        discrete_state = get_discrete_state(env, env.reset(),DISCRETE_OS_SIZE)
        
        # if discrete_state >= 19: # This only works for the specific case where there are 20 indices
        #     discrete_state = 19


        if episode % EPISODE_EVALUATION == 0:
            # print(f"Episode {episode}")
            print("Episode:", episode)
            render = True        
        else:
            render = False
        while not done:
            values = q_table[discrete_state]
            action = evaluation.get_action_egreedy(values,EPSILON)
            # action = np.argmax(q_table[discrete_state]) # Definitely greedy approach
            new_state, reward, done, _ = env.step(action)
            # print(f"Rewards {reward}")
            new_discrete_state = get_discrete_state(env, new_state,DISCRETE_OS_SIZE)
            if render:
                env.render()

            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action, )]
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_RATE * max_future_q)
                q_table[discrete_state + (action, )] = new_q # This was the critical step. You are updating the current Q state, not the future one!
            elif reward == 0:
            # elif new_state[0] >= env.goal_position: #env.goal_position = 0.5 for MountainCar-v0 environment # What about acrobot?
                q_table[new_discrete_state + (action, )] = 0 # Automatically, this is an optimal state!
                print("Done at Episode: ", episode)
            discrete_state = new_discrete_state         

    env.close()
    return q_table

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Choose the environment we are working with')
    parser.add_argument(
        '--car', help = "Choose the MountainCar gym environment", action = "store_true")
    parser.add_argument(
        '--maze', help = "Choose the Maze environment", action = "store_true"
    )
    parser.add_argument(
        '--acrobot', help = "Choose the Acrobat environment", action = "store_true"
    )
    args = parser.parse_args()
    if args.car:
        env = gym.make("MountainCar-v0")
        EPISODES = 15000
        name = "Q_car"
        Q = train_gym(env, EPISODES)
        np.save(name,Q)

    elif args.maze:
        name = "Q_maze"
        EPISODES = 5000
        maze_env = MDP()
        q_table = maze_env.QL_train_maze()
        # for episode in EPISODES:

        np.save(name,q_table)
        
        pass
    elif args.acrobot:
        name = "Q_acrobot"
        EPISODES = 15000
        env = gym.make("Acrobot-v1")
        Q = train_gym(env, EPISODES)
        np.save(name,Q)



    
    
    
        