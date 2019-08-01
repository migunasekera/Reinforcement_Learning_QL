import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import optim
import torch
import gym
from evaluation import evaluate
import matplotlib.pyplot as plt
import time
import argparse
from maze import Maze
import matplotlib.pyplot as plt
import time

class PolicyEstimator(nn.Module):
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
    This will take the reward, and recursively update them, in reverse.
    
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
    # May cause conflicts with the Maze environment
    action_probs = policy_estimator(state).detach().numpy()
    action = np.random.choice((policy_estimator.n_output), p = action_probs)
    idx = action_probs[action]
    return action, idx

def reward_shaper(env, state):
    '''
    based on Andrew Ng's work: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
    
    Purpose is to introduce less sparse, smaller rewards, so that reaching the next bigger reward is more attainable
    
    This is designed for the MountainCar example, which has a very sparse reward. In fact, it may be better to introduce something that would lead to a positive reward at its max!
    
    Normalize to be between 0 and 1
    '''
    return np.abs(state) / env.observation_space.high


def generate_episode(env, policy_estimator):
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


def reinforce(env, policy_estimator, EPISODES = 2000, isBaseline = False, lr = .001, gamma = 0.99):
    '''
    When going through episodes, this will do reward shaping
    
    '''
    cumulative_reward = []
    optimizer = optim.Adam(pe.parameters(), lr = lr)

    for episode in range(EPISODES):
        cum_rewards, reward_list, action_list, state_list = generate_episode(env, policy_estimator)
        
        # Use the cumulative reward metric in the end. Now that I've added reward shaping, it would be best to show whether the end goal converges, rather than what I'm shaping
        cumulative_reward.append(cum_rewards)

        # Compute loss
        reward_tensor = torch.tensor(reward_list).float()
        action_tensor = torch.tensor(action_list).long()
        state_tensor = torch.stack(state_list)

        # Does this need to be seeded?
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
        # print(return_tensor[-10:])

        # loss has negative sign, because we are doing gradient ascent
        loss = torch.sum(-return_tensor * picked_logprob)
    #     loss_list.append(loss)
        print(f'\r episode {episode} -------- loss {loss} -------- rewards {cumulative_reward[-1]}', end = " ")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return cumulative_reward
           
                
if __name__ == "__main__":
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

    args = parser.parse_args()


EPISODES = 2000
gamma_list = [0.99, 0.90]
# lr_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


fig, axs = plt.subplots(1,1, figsize = (15,11))


for g in gamma_list:
#     env = gym.make("Acrobot-v1")
    
    for run in range(RUNS):
    
        if args.car:
            env = gym.make("MountainCar-v0")
            env.max_episode_steps = 1000
        if args.maze:
            # This doesn't have seed argument, so this could break it most likely
            env = Maze()
        if args.acrobot:
            env = gym.make("Acrobot-v1")
        if args.cart:
            env = gym.make("CartPole-v0")
        seed = env.seed()
        seeds.append(seed)
    env.seed(0)
    pe = PolicyEstimator(env)
    
    start = time.time()
    rewards = reinforce(env, pe, isBaseline = True, gamma = g)
    print(time.time() - start)
    window = 10
    smoothed_rewards = [np.mean(rewards[i-window:i+1]) if i > window 
                            else np.mean(rewards[:i+1]) for i in range(len(rewards))]
    axs.plot(smoothed_rewards)

axs.set_xlabel("Episodes")
axs.set_ylabel("Rewards")
axs.legend(gamma_list)
fig.suptitle(f"{env.spec._env_name} rewards on vanilla REINFORCE with moving average baseline with reward shaping", fontsize = 16)
plt.show()

