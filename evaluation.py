# Heejin Chloe Jeong

import numpy as np
import torch



def get_action_egreedy(values ,epsilon):
	# Implement epsilon greedy action policy
	if np.random.random() < epsilon:
		return np.random.randint(0,len(values))
	else:
		return np.argmax(values)
	NotImplementedError

def get_action_policy_grad(policy_estimator, state):
	action_probs = policy_estimator(state).detach().numpy()
	action = np.random.choice((policy_estimator.n_output), p = action_probs)

	return action
    

def evaluate(env, Q_table = None, step_bound = 100, num_itr = 10, Gym = False, policy_estimator = None):
	"""
	evaluation for discrete state and discrete action spaces and an episodic environment.
    
    If it's not a 

	Input:
		env : an environment object. 
		Q : A numpy array. Q values for all state and action pairs. 
			Q.shape = (the number of states, the number of actions)
		step_bound : the maximum number of steps for each iteration
		num_itr : the number of iterations

	Output:
		Total number of steps taken to finish an episode (averaged over num_itr trials)
		Cumulative reward in an episode (averaged over num_itr trials)

	"""
	total_step = 0 
	total_reward = 0 
	itr = 0 
	if Gym:
		step_bound = 200
	while(itr<num_itr):
		step = 0
		np.random.seed()
		state = env.reset()



		reward = 0.0
		done = False
		while((not done) and (step < step_bound)):
			
			if Gym:
				if Q_table is not None:
					discrete_state = get_discrete_state(env, state)
					action = get_action_egreedy(Q_table[discrete_state], 0.05)
					state_n, r, done, _ = env.step(action)
				
				elif policy_estimator is not None:
					state = torch.FloatTensor(state)
					action = get_action_policy_grad(policy_estimator, state)
					
					# action = get_action_policy_grad(policy_estimator, state)
					state_n, r, done, _ = env.step(action)
			else:
				action = get_action_egreedy(Q_table[state], 0.05)
				r, state_n, done = env.step(state,action)
			state = state_n
			reward += r
			step +=1
		total_reward += reward
		total_step += step
		itr += 1
	return total_step/float(num_itr), total_reward/float(num_itr)

def get_discrete_state(env, state):
    
	DISCRETE_OS_SIZE = [20] * len(env.observation_space.high) # Generates the size of observations table. 20 was randomly chosen
	discrete_window_size = (env.observation_space.high - env.observation_space.low)  / DISCRETE_OS_SIZE # Step sizes!
	# fudge_factor = (0.01) * env.observation_space.low # This doesn't work b/c the number could be negative
	discrete_state = (state - env.observation_space.low) / discrete_window_size
	# print(discrete_state)
	return tuple(discrete_state.astype(np.int))
