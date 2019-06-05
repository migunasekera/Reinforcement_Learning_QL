# Heejin Chloe Jeong

import numpy as np



def get_action_egreedy(values ,epsilon):
	# Implement epsilon greedy action policy
	if np.random.random() < epsilon:
		return np.random.randint(0,len(values))
	else:
		return np.argmax(values)
	NotImplementedError

def evaluation(env, Q_table, step_bound = 100, num_itr = 10, Gym = False):
	"""
	Semi-greedy evaluation for discrete state and discrete action spaces and an episodic environment.

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
				discrete_state = get_discrete_state(env, state)
				action = get_action_egreedy(Q_table[discrete_state], 0.05)
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
