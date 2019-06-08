import gym
import time
import sys
import numpy as np
import evaluation
import argparse
from maze import *
from evaluation import get_action_egreedy, evaluation
import matplotlib.pyplot as plt


class MDP(Maze):
    def __init__(self):
        super().__init__()
        # self.LEARNING_RATE = 0.1
        

def RMSE_calc(Q_t, Q_opt):
    '''
    Root Mean square error: 
    '''
    return np.sqrt(np.sum((Q_opt - Q_t) ** 2) / len(Q_opt.flatten()))

def RMSE(Q_t, Q_opt):
    '''
    Root Mean square error: 
    '''
    return np.sqrt(np.sum((Q_opt - Q_t) ** 2) / len(Q_opt.flatten()))




######################################################################################################
#Maze
######################################################################################################
def train_maze(env, EPISODES, EPISODE_EVALUATION, EPSILON, LEARNING_RATE, findOptimal = False):
    '''
    env: Maze environment
    EPSILON: Determines epsilon for epsilon greedy algorithm. Higher the value, the less greedy [0,1]
    EPISODE_EVALUATION: Determines the number of episodes before we will evaluate how well the maze is doing
    EPISODES: Number of episodes that the learning algorith will run
    '''
    name = "Maze"
    DISCOUNT_RATE = 0.95
    tmpEval = []
    q_table  = np.random.uniform(low = 0, high = 3, size = (env.snum,env.anum))

    print(findOptimal)

    if findOptimal is False:
        q_optimal = np.load('Q_maze.npy', allow_pickle = True)
    
    else:

        # Throwaway thing, just to speed up the coding. Won't actually use it, but I don't want to use logic below in RMSE calculation
        q_optimal = np.copy(q_table)
    RMSE = []

    
    print(q_table.shape)

 
    for episode in range(EPISODES):
        done = False
        state = env.reset()
        
        # Stop training to do an evaluation
        if episode % EPISODE_EVALUATION == 0:
            print(f"\nEpisode {episode}")
            avg_step, avg_reward = evaluation(env,q_table)
            print(f"Avg Step: {avg_step} \nAvg Reward: {avg_reward}")
            tmpEval.append((episode,avg_step,avg_reward))
            if findOptimal:
                print(f"Saving episode {episode}...")
                np.save(f"Optimal/{name}/{episode}.npy",q_table)
            
            time.sleep(1)
            continue


        while not done:
            values = q_table[state]
            action = get_action_egreedy(values, EPSILON)
            reward, new_state, done = env.step(state, action)


            if not done:
                max_future_q = np.max(q_table[new_state])
                current_q = q_table[(state,action)]
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_RATE * max_future_q)
                q_table[(state,action)] = new_q # This was the critical step. You are updating the current Q state, not the future one!
            elif done: # Not too sure about how to give reward on the maze, and what the condition should be
                q_table[(new_state,action)] = reward ##Not too sure what o do here

                RMSE.append((episode,RMSE_calc(q_table, q_optimal)))

                # Automatically, this is an optimal state! This is the recursive definition at the end goal I believe.
            state = new_state

    
    
    realEval = np.array(tmpEval)

    RMSE = np.array(RMSE)
    # print(RMSE.shape)
 




    return q_table, realEval, RMSE


def evaluation_plot(realEval, RMSE, *args, save = False):
    fig, axs = plt.subplots(1,3)
    fig.suptitle(f"Evaluation metrics for the {args[0]} puzzle with current Q-Learning strategy")

    episodes = realEval[...,0]
    steps = realEval[...,1]
    reward = realEval[...,2]
    RMSE_val = RMSE[...,1]


    axs[0].plot(realEval[...,0],realEval[...,1])
    axs[0].set_xlabel("Episodes")
    axs[0].set_ylabel("Steps")
    axs[0].set_xlim((0,EPISODES))
    axs[0].set_ylim((0,np.max(steps)))
    axs[0].set_title("Number of Steps")
    #TODO: Implement curve fitting, to make the plots more reasonable

    axs[1].plot(realEval[...,0],realEval[...,2])
    axs[1].set_xlabel("Episodes")
    axs[1].set_ylabel("Reward")
    axs[1].set_xlim((0,EPISODES))
    axs[1].set_ylim((np.min(reward),np.max(reward)))
    axs[1].set_title("Reward (0.0 - 3.0)")


    axs[2].plot(RMSE[...,0],RMSE[...,1])
    axs[2].set_xlabel("Episodes")
    axs[2].set_ylabel("RMSE")
    axs[2].set_xlim((0,EPISODES))
    axs[2].set_ylim((0,np.max(RMSE_val)))
    axs[2].set_title("Root mean squared error")
    if save:
        fig.savefig(f"Results/{args[0]}_{int(args[1])}_{int(args[2])}.png")

    plt.show()
    print("Finished")

######################################################################################################
#GYM
######################################################################################################
def get_discrete_state(env, state, DISCRETE_OS_SIZE):
    
    discrete_window_size = (env.observation_space.high - env.observation_space.low)  / DISCRETE_OS_SIZE # Step sizes!
    # fudge_factor = (0.01) * env.observation_space.low # This doesn't work b/c the number could be negative
    discrete_state = (state - env.observation_space.low) / discrete_window_size
    # print(discrete_state)
    return tuple(discrete_state.astype(np.int))


def train_gym(env, EPISODES, EPISODE_EVALUATION, EPSILON, LEARNING_RATE, findOptimal = False):
    '''
    Input: Env: Gym environment

    This will train a vlaue 
    '''

    # LEARNING_RATE = 0.1
    DISCOUNT_RATE = 0.95
    # EPSILON = 0.10
    # EPISODE_EVALUATION = 500 #Evaluate the current policy at these times. Plottable
        


    
    
    DISCRETE_OS_SIZE = [20] * len(env.observation_space.high) # Generates the size of observations table. 20 was randomly chosen
 
    q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))
    if findOptimal is False:
        if env.observation_space.shape[0] == 2: ## Car environment
            q_optimal = np.load('Q_car.npy', allow_pickle = True) # This is only for the car environemnt
        elif env.observation_space.shape[0] == 6: ## Acrobot environment
            q_optimal = np.load("Q_acrobot.npy")
    
    else:
        # Throwaway thing, hacky way to solve it actually
        if env.observation_space.shape[0] == 2: ## Car environment
            name = "Car"
        elif env.observation_space.shape[0] == 6: ## Acrobot environment
            name = "Acrobot"
        q_optimal = np.copy(q_table)
    



   
    tmpEval = []
    RMSE = []
    # RMSE = []

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
            print(f"\nEpisode {episode}")
            avg_step, avg_reward = evaluation(env,q_table,Gym = True)
            print(f"Avg Step: {avg_step} \nAvg Reward: {avg_reward}")
            tmpEval.append((episode,avg_step,avg_reward))
            time.sleep(1)
            discrete_state = get_discrete_state(env, env.reset(),DISCRETE_OS_SIZE)
            if findOptimal:
                print(f"Saving episode {episode}...")
                np.save(f"Optimal/{name}/{episode}.npy",q_table)
            render = True
        else:
            render = False
        while not done:
            values = q_table[discrete_state]
            action = get_action_egreedy(values,EPSILON)
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
                #################################################
                #FIX THIS BUG
                #################################################
                
                print("Done at Episode: ", episode)
            discrete_state = new_discrete_state
        # print(f"This is episode {episode}")
        if findOptimal is False:
            RMSE.append((episode, RMSE_calc(q_table,q_optimal)))
    realEval = np.array(tmpEval)
    RMSE = np.array(RMSE)

    env.close()
    return q_table, realEval, RMSE

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
    parser.add_argument(

        '--optimal', help = "Will run to create a Q-table that will be saved as an npy format. This is the optimal q-value that will be compared against", action = "store_true"

    )
    parser.add_argument(

        '--save', help = "Will save the value that is computed, for whatever learning algorithm you chose", action = "store_true"

    )

    args = parser.parse_args()




    if args.car:
        env = gym.make("MountainCar-v0")
        EPISODES = 15000
        EPISODE_EVALUATION = 500
        EPSILON = 0.10
        LEARNING_RATE = 0.1
        
        name = "Q_car"
        if args.optimal:
            Q, real_eval, RMSE = train_gym(env, EPISODES, EPISODE_EVALUATION, EPSILON, LEARNING_RATE, findOptimal= True)
            evaluation_plot(real_eval, RMSE, name, EPSILON * 100, LEARNING_RATE * 100)
            # np.save(name,Q)
            # after 12000, it's pretty good
        
        elif args.save:
            Q, real_eval, RMSE = train_gym(env, EPISODES, EPISODE_EVALUATION, EPSILON, LEARNING_RATE)
            evaluation_plot(real_eval, RMSE, name, EPSILON * 100, LEARNING_RATE * 100, save = True)
        else:
            Q, real_eval, RMSE = train_gym(env, EPISODES, EPISODE_EVALUATION, EPSILON, LEARNING_RATE)
            evaluation_plot(real_eval, RMSE, name, EPSILON * 100, LEARNING_RATE * 100)

        

    elif args.maze:
        name = "Q_maze"
        EPISODES = 5000

        EPISODE_EVALUATION = 200
        EPSILON = 0.10
        LEARNING_RATE = 0.1
        env = MDP()
        # for episode in EPISODES:
        if args.optimal:
            Q, real_eval, _ = train_maze(env, EPISODES, EPISODE_EVALUATION, EPSILON, LEARNING_RATE, findOptimal= True)
            evaluation_plot(real_eval, RMSE, name, EPSILON * 100, LEARNING_RATE * 100)
            # np.save(name,Q)
        elif args.save:
                Q, real_eval, RMSE = train_maze(env, EPISODES, EPISODE_EVALUATION, EPSILON, LEARNING_RATE)
                evaluation_plot(real_eval, RMSE, name, EPSILON * 100, LEARNING_RATE * 100, save = True)
        else:
            Q, real_eval, RMSE = train_maze(env, EPISODES, EPISODE_EVALUATION, EPSILON, LEARNING_RATE)
            evaluation_plot(real_eval, RMSE, name, EPSILON * 100, LEARNING_RATE * 100)
            # 3000 was a good one, w


        pass
    elif args.acrobot:
        name = "Q_acrobot"
        EPISODES = 15000
        EPISODE_EVALUATION = 500
        EPSILON = 0.10
        LEARNING_RATE = 0.1
        env = gym.make("Acrobot-v1")

        if args.optimal:
            Q, real_eval, _ = train_gym(env, EPISODES, EPISODE_EVALUATION, EPSILON, LEARNING_RATE, findOptimal= True)
            evaluation_plot(real_eval, RMSE, name, EPSILON * 100, LEARNING_RATE * 100)
            # np.save(name,Q)
        
        elif args.save:
            Q, real_eval, RMSE = train_gym(env, EPISODES, EPISODE_EVALUATION, EPSILON, LEARNING_RATE)
            evaluation_plot(real_eval, RMSE, name, EPSILON * 100, LEARNING_RATE * 100, save = True)
        else:
            Q, real_eval, RMSE = train_gym(env, EPISODES, EPISODE_EVALUATION, EPSILON, LEARNING_RATE)
            evaluation_plot(real_eval, RMSE, name, EPSILON * 100, LEARNING_RATE * 100)



    
    
    
        