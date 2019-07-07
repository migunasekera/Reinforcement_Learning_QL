import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import optim
import torch
import gym
from evaluation import evaluate
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
    
def action_choice(policy_estimator, state):   
    action_probs = policy_estimator(state).detach().numpy()
    action = np.random.choice((policy_estimator.n_output), p = action_probs)
    return action

def reinforce(env, policy_estimator, DISCOUNT = 0.99, EPISODES = 2000, lr = 0.01, EVALUATION_STEP = 400):
    optimizer = optim.Adam(policy_estimator.parameters(), lr = lr)
    
    # Generate episodes
    

    
    cumulative_reward = []
    running_reward = 0
    
    steps = []
    eval_reward = []
        
    for ep in range(EPISODES):
        
        if ep % EVALUATION_STEP == 0:
            st, er = evaluate(env, Q_table = None, step_bound = 100, num_itr = 10, Gym = True, policy_estimator = policy_estimator)
            steps.append(st)
            eval_reward.append(er)
        
        # Record actions, states, and rewards for each episode
        states = []
        actions = []
        rewards = []
        total_rewards = []
        done = False
            
   
        
        # If it isn't the x episode, then continue the evaluation
        
        s_0 = torch.FloatTensor(env.reset())
    
        while done is False:
            
            action = action_choice(policy_estimator, s_0)
            
#             action_probs = policy_estimator(s_0).detach().numpy()
#             action = np.random.choice((policy_estimator.n_output), p = action_probs)

            s_1, reward, done, _ = env.step(action)
            
            states.append(s_1)
            actions.append(action)
            rewards.append(reward)
            
            if done:
                
                G = discount_rewards(rewards)
                
                action_tensor = torch.LongTensor(actions)
                states_tensor = torch.tensor(states).float()
#                 reward_tensor = torch.tensor(rewards).float()
                reward_tensor = torch.tensor(G).float()
#                 total_rewards.append(sum(rewards))
                
                
                
                
                # collect them all. convert reward per step into G. 
                
                running_reward += reward_tensor.sum()
                cumulative_reward.append(running_reward.item())
                
                #Create the loss function
                #logprobs = torch.log(G * policy_estimator(states_tensor))
                 # Basically create a tensor size [1, len(reward_tensor)] Below doesn't actually work
                WINDOW = 20

#                 baseline = np.mean(reward_tensor[-WINDOW:].detach().numpy())


#                 # Calculate loss
#                 reward_with_baseline = reward_tensor - baseline
                
                
                
                logprob = torch.log(policy_estimator(states_tensor))
#                 selected_logprobs = reward_with_baseline * logprob[np.arange(len(action_tensor)), action_tensor]
                selected_logprobs = reward_tensor * logprob[np.arange(len(action_tensor)), action_tensor]
                loss = -selected_logprobs.sum()

                    
        
    
#     logprobs = torch.log(G * action_tensor)
#                 loss = -torch.sum(logprobs)
                
#                 print(f'running_reward {running_reward} loss: {loss}')
                print("\r Ep {}/{} running reward: {}, loss {}".format(ep, EPISODES, running_reward, loss), end = "")
                #Backprop
                optimizer.zero_grad()
                loss.backward()
                
                
                optimizer.step()
                
                states = []
                actions = []
                rewards = []
                
    
                
                      
                      
              
                
                
                
        # Show the results of the neural net (inference step)
        
        
    return cumulative_reward, steps, eval_reward
                
           
                
if __name__ == "__main__":
    RUNS = 2 # number of different seeds you are trying out - checking by eye for variance


fig, axs = plt.subplots(RUNS,1,figsize= (15 * RUNS,11))
seeds = []




for run in range(RUNS):
    env = gym.make("CartPole-v0")
#     env._max_episode_steps = 1000
    seed = env.seed()
    seeds.append(seed)
    pe = PolicyEstimator(env)
    
    
    start = time.time()
    cum_reward, steps, eval_reward = reinforce(env, pe, EPISODES = 2000)
    
    end = time.time()
    print(f'time of run {run} for seed {seed} in minutes: {(end-start) / 60.}')
    axs[0].plot(steps)
    axs[0].set_ylabel("Steps taken")
    axs[0].set_xlabel("Episodes")
    axs[0].legend(seeds)
    
    print(f'time of run {run} for seed {seed} in minutes: {(end-start) / 60.}')
    axs[1].plot(eval_reward)
    axs[1].set_ylabel("Cumulative reward")
    axs[1].set_xlabel("Episodes")
    axs[1].legend(seeds)
    plt.show()



