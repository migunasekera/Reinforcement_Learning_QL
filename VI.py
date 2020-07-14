from QL import evaluation_plot
import numpy as np
import matplotlib.pyplot as plt
import pprint
import sys
from maze import Maze
import argparse
if "../" not in sys.path:
  sys.path.append("../") 

pp = pprint.PrettyPrinter(indent=2)

class MDP(Maze):
    '''
    Inherits all the attributes from the maze
    '''
    def __init__(self):
        super().__init__()
        # self.LEARNING_RATE = 0.1

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.
    
    Parameters:
    -----------
        env: Maze class, which is inherited from the Maze function that was given for the assignment.
        Importantly, it has the following attributes:
            env.snum is a number of states in the environment. 
            env.anum is a number of actions in the environment.
        And the following function which is utilized:
            env.step is a list of transition tuples (reward, next_state, done)
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

      
    Returns:
    --------
    A tuple (policy, V, Q_table) of the optimal policy, optimal value function, and optimal action-value table

    """
    

    V = np.random.random((env.snum, )) # Arbitrarily initialize V(s)
    V[-1] = 0 #V(terminal is zero)
    policy = np.ones([env.snum, env.anum])
    
    
    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.
        
        Parameters:
        -----------
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.snum
        
        Returns:
        --------
            A vector of length env.anum containing the expected value of each action.
        """
        A = np.zeros(env.anum)
        for a in range(env.anum):
            reward, next_state, _  = env.step(state,a)
            A[a] += (reward + discount_factor * V[next_state])
        return A
    
    loop = 0
    


    # Implement!
    while True:
        delta = 0
        for state in range(env.snum):
            v = V[state]
            A = one_step_lookahead(state, V)
            
            V[state] = np.max(A)
            
            print(f'state {state} delta {delta} V[state] {V[state]} Action {A}')
#             print(V[state])
            delta = np.max((delta, v - V[state]))
        # print(delta)
        print(f'Loop: {loop}, delta: {delta}')
        if delta < theta:
            break
        loop += 1
    # Fill in the policy
    
    A = np.zeros(env.anum)
    # Does value iteration create an optimal Q-table?

    Q_table = np.zeros((env.snum, env.anum))
    for state in range(env.snum):
        A = one_step_lookahead(state,V)
        Q_table[state] = A
        tmp = np.zeros(policy[state].shape)
        idx = np.argmax(A)
        tmp[idx] = 1
        policy[state] = tmp
    
    
    return policy, V, Q_table

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(
        description='Choose the environment we are working with')
    
    parser.add_argument(

        '--optimal', help = "Will run to create a Q-table that will be saved as an npy format. This is the optimal q-value that will be compared against", action = "store_true"

    )
    
    args = parser.parse_args()
    env = MDP()

    policy, v, Q_table = value_iteration(env, theta=0.0001, discount_factor=1.0)
    
    print(policy)
    print("")


    print("Value Function:")
    print(v)
    print("")

    print("Q-table:")
    print(Q_table.shape)
    print("")

    if args.optimal:
        np.save("Optimal/Maze/VI/Q_table.npy", Q_table)
