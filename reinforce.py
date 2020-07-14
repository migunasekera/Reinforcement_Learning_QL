import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import optim
import torch
import gym
from gym import wrappers
from evaluation import evaluate
import matplotlib.pyplot as plt
import time
import argparse
from maze import Maze
import matplotlib.pyplot as plt
import time
import pickle as pl

class PolicyEstimator(nn.Module):
    ''' 
    Feedforward Neural Net written in PyTorch, to model Policy Gradient estimator

    model:
    - 1 hidden layer
    - 16 nodes
    - ReLU activation

    input: Gym environment observation (state) space
    output: Gym environment action space probabilities based on Softmax function

    '''
    def __init__(self, env):
        super(PolicyEstimator,self).__init__()
        
        self.n_input = env.observation_space.shape[0]
        self.n_output = env.action_space.n
        self.fc1 = nn.Linear(self.n_input, 16)
        self.fc2 = nn.Linear(16, self.n_output)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim = -1)
        return x

def discount_rewards(reward_array, DISCOUNT = 0.99):
    '''
    Returns a reward function as part of loss function, which is updated recursively in reverse in Monte-Carlo fashion
    
    parameters:
    -----------
    reward_array: array of rewards delivered for each time step in episode
    DISCOUNT: discounting value, known as gamma. Multiplier for reward lookup

    returns:
    --------
    G: array value on top 
    
    '''
    
#     reward_array = torch.FloatTensor(reward_array)
    G = torch.zeros((len(reward_array),)).float()
    cumulative = 0
    for i in reversed(range(len(reward_array)-1)):
        G[i] = reward_array[i+1] + DISCOUNT * G[i+1]
    return G

# def discount(reward, DISCOUNT = 0.99):
    
# def action_choice(policy_estimator, state):   
#     action_probs = policy_estimator(state).detach().numpy()
#     action = np.random.choice((policy_estimator.n_output), p = action_probs)
#     return action

def action_choice(policy_estimator, state):
    '''
    Given a state, an Action is chosen via uniform sampling from a probability distribution from the Policy estimator. This function assumes a discrete action space

    parameters:
    -----------
    policy_estimator: Function approximator based on Feedforward NN model
    state: observation (state) vector in Gym environment observation space

    returns:
    --------
    action: value in action space chosen from policy gradient


    '''
    # May cause conflicts with the Maze environment
    action_probs = policy_estimator(state).detach().numpy()
    action = np.random.choice((policy_estimator.n_output), p = action_probs)
    idx = action_probs[action]
    return action, idx

def reward_shaper(env, state):
    '''
    based on Andrew Ng's work: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
    
    Purpose is to introduce less sparse, smaller rewards, so that reaching the next bigger reward is more attainable This is designed for the MountainCar example, which has a very sparse reward. In fact, it may be better to introduce something that would lead to a positive reward at its max!

    returns:
    --------
    Value that is normalized between between 0 and 1
    '''
    return np.abs(state) / env.observation_space.high


def generate_episode(env, policy_estimator, renderEpisode = False):
    '''
    parameters:
    -----------
    env: gym environment
    policy_estimator: NN model to generate policy decisions
    isRendered: If rendered, then have the model render at specified times (in the code, currently set to every 10 episodes)

    returns:
    --------
    cumulative_reward (int): total reward delivered in episode
    reward_list (list[int]): reward delievered at each time step
    action_list (list[int]: action taken at each time step
    state_list (list[float]): state at each time step
    '''
    done = False




    action_prob_val = []
    action_list = []
    state_list = []
    s_0 = torch.from_numpy(env.reset()).float() # Initial state
    cumulative_reward = 0
    reward_list = []


    while not done:
        # Pick an action, and step through environment
        
        action, prob = action_choice(pe, s_0)

        # Append this before you do reward shaping. This will be used as comparison
        state_list.append(s_0)

        s_1, reward, done, _ = env.step(action)

        # Render the episode
        if renderEpisode == True:
            env.render()
        
        # Track this on the graph, but use reward shaping in your calculations
        cumulative_reward += reward
        ######################################################################################
#       reward shaping
        if env.spec._env_name == 'MountainCar':
            
            # Arbitrary value just to see how results look like. I purposely made it less than 1
            
            

            beta = 1.25 # This is the key. For Mountaincar, this needs make it net positive if you reach maximum velocity. Since all rewards are -1, at max velocity this would give a positve reward
            F = reward_shaper(env, s_1) # Reward shaping function
            reward = reward + beta * F[1]
        ######################################################################################
            
            
        
        action_prob_val.append(prob)
        reward_list.append(reward)
        action_list.append(action)
        s_0 = torch.from_numpy(s_1).float() # Make state the next one
    return cumulative_reward, reward_list, action_list, state_list


def reinforce(env, policy_estimator, EPISODES = 2000, isBaseline = False, lr = .001, gamma = 0.99, isRendered = False):
    '''
    Runs REINFORCE algorithm. Reinforce is a policy gradient algorithm which creates a loss function to perform gradient ascent, which maximizes rewards by exploring parameter space

    params:
    -------
    env: gym environment
    policy_estimator:
    EPISODES: number of episodes simulation is run through
    isBaseline (bool): decide whether baseline is used or not
    lr: learning rate of Adam optimizer
    gamma: discount factor [0,1]

    returns:
    --------
    cumulative rewards (list): list of numerical reward per episode

    
    '''
    recordTimes = {1,5,50,150,300,500,1000,1999}
    cumulative_reward = []
    optimizer = optim.Adam(pe.parameters(), lr = lr)
    

    for episode in range(EPISODES):

        if isRendered is True:
                # render this run if selected
                if episode in recordTimes:
                    generate_episode(env, policy_estimator, renderEpisode = True)

        # This run will be used to update hyperparameters
        cum_rewards, reward_list, action_list, state_list = generate_episode(env, policy_estimator, renderEpisode = False)

        
        cumulative_reward.append(cum_rewards)

        # Compute loss
        reward_tensor = torch.tensor(reward_list).float()
        action_tensor = torch.tensor(action_list).long()
        state_tensor = torch.stack(state_list)

        action_output = torch.log(policy_estimator(state_tensor))

        # This is just doing an indexing! It picks the action that was chosen in the run. There is no operation here that pytorch has to track
        picked_logprob = action_output[np.arange(len(action_tensor)), action_tensor]

        # Adds a return value (with discounting) for every time point until episode finishes e.g [-200, -198, ......, 0]
        return_tensor = discount_rewards(reward_tensor, DISCOUNT = gamma)
        
        if isBaseline:
            WINDOW = 20          
            
            # ----- BASELINE ----- #
            baseline = torch.stack([torch.mean(return_tensor[i-WINDOW:i+1]) if i > WINDOW 
                        else torch.mean(return_tensor[:i+1]) for i in range(len(return_tensor))])
            return_tensor = return_tensor - baseline

        # loss has negative sign, because we are doing gradient ASCENT
        print(f'\r episode {episode} -------- loss {loss} -------- rewards {cumulative_reward[-1]}', end = " ")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return cumulative_reward






if __name__ == "__main__":
    # ------------------------------------PARAMETERS ------------------------------------------------ #
    parser = argparse.ArgumentParser(
        description='Choose the environment we are working with. All will be run over 2000 episodes')
    parser.add_argument(
        '--car', help = "Choose the MountainCar gym environment", action = "store_true")
    parser.add_argument(
        '--maze', help = "Choose the Maze environment", action = "store_true"
    )
    parser.add_argument(
        '--cart', help = 'Choose the CartPole environment', action = "store_true"
    )
    parser.add_argument(
        '--acrobot', help = "Choose the acrobot environment", action = "store_true"
    )

    parser.add_argument(
        '--baseline', help = "tell that there's a baseline", action = "store_true"
    )
    parser.add_argument(
        '--render', help = "render at specific time intervals set in REINFORCE", action = "store_true"
    )

    args = parser.parse_args()


    # -------------- PARAMETERS ------------------ #
    EPISODES = 2000
    gamma_list = [0.99, 0.90]


    fig, axs = plt.subplots(1,1, figsize = (15,11))

    include_baseline = [False, True]
    if args.baseline:
        includeBaseline = True 
    else:
        includeBaseline = False

    if args.render:
        includeRender = True 
    else:
        includeRender= False

    for g in gamma_list:
        
        if args.car:
            env = gym.make("MountainCar-v0")
            env.max_episode_steps = 1000
        elif args.maze:
            # This doesn't have seed argument, so this could break it most likely
            env = Maze()
        elif args.acrobot:
            env = gym.make("Acrobot-v1")
        elif args.cart:
            env = gym.make("CartPole-v0")
#        else:
#            raise NameError("No environment specified!")

        env.seed(0)
        pe = PolicyEstimator(env)
        start = time.time()
        rewards = reinforce(env, pe, isBaseline = includeBaseline, gamma = g, isRendered = includeRender)



        print(time.time() - start)
        window = 10
        smoothed_rewards = [np.mean(rewards[i-window:i+1]) if i > window 
                                else np.mean(rewards[:i+1]) for i in range(len(rewards))]
        axs.plot(smoothed_rewards)

    axs.set_xlabel("Episodes")
    axs.set_ylabel("Rewards")
    fig.suptitle(f"{env.spec._env_name} baseline vs. no baseline on REINFORCE", fontsize = 16)
    plt.show()
    with open('pickled_figures/acrobot_baselineComparison_99.pickle','wb') as fid:
        pl.dump(fig, fid)

