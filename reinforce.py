import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import optim
import torch
import gym
import evaluation as ev


env = gym.make("MountainCar-v0")

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
    # state = torch.FloatTensor(state)

    
    action_probs = policy_estimator(state).detach().numpy()
    # Probability may come out as NaN
    action = np.random.choice((policy_estimator.n_output), p = action_probs)
    
    return action

def reinforce(env, policy_estimator, DISCOUNT = 0.99, EPISODES = 2000, lr = 0.01, EVALENV = 500):
    optimizer = optim.Adam(policy_estimator.parameters(), lr = lr)
    
    # Generate episodes
    

    
    running_reward = 0
    for ep in range(EPISODES):
        
        # Record actions, states, and rewards for each episode
        states = []
        actions = []
        rewards = []
        done = False
        
    
        if ep % EVALENV == 0:
            ev.evaluation(env, Q_table = None, step_bound = 100, num_itr = 10, Gym = True, policy_estimator = net)

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
                reward_tensor = torch.tensor(rewards).float()
                
                
                
                
                # collect them all. convert reward per step into G. 
                
                running_reward += reward_tensor.sum()
                
                #Create the loss function
                #logprobs = torch.log(G * policy_estimator(states_tensor))
                logprob = torch.log(policy_estimator(states_tensor))
                selected_logprobs = reward_tensor * logprob[np.arange(len(action_tensor)), action_tensor]
                loss = -selected_logprobs.mean()

                    
        
    
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
        
        
    return policy_estimator
           
                
if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    net = PolicyEstimator(env)
    optimized_net = reinforce(env, net)
    np.save('REINFORCE-1')



