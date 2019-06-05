from maze import *
from vplot import *
import numpy as np
import math


# env = Maze()
# initial_state = env.reset()
# state = initial_state
# action = np.random.choice(4,)
# reward, next_state, done = env.step(state, action)
# print(next_state)
# env.plot(state, action)

discount = 0.9
iters = 1000

class MDP(Maze):
    def __init__(self):
        super().__init__()


    def calcQ(self, state, action):
        Vsum = 0
        rllist = []
        slip_action = ACTMAP[action]
        slip_reward, slip_next_state, _ = super().step(state,slip_action, slip = False)
        rllist.append((slip_reward, slip_next_state, self.slip))
        
        rllist.append((reward, next_state, 1 - env.slip))

        for reward, next_state, pi in rllist:
            Vsum += math.pi * (reward + discount * values[next_state])
        return Vsum


if __name__ == "__main__":
    env = MDP()
    values = np.zeros((env.snum))
    Qval = np.zeros((env.snum, env.anum))
    optpolicies = np.zeros(env.snum)


    # Value iteration

    for i in range(iters):
        tmpV  = np.zeros(env.snum)
        for state in range(env.snum):
            if env.idx2cell[int(state/8)] == env.goal_pos:
                continue
            Vmax = float('-inf')
            for action in range(env.anum):
                Vsum = env.calcQ(state, action)
                Vmax = max(Vmax, Vsum) # Determines the value of the function
            tmpV[state] = Vmax
        values = np

    for state in range(env.snum):
        for action in range(env.anum):
            Qval[state, action] = env.calcQ(state, action)

    for state in range(env.snum):
        optpolicies[state] = np.argmax(Qval[state,:])


    np.save('Results/Q_values',Qval)
    print(Qval)
    #np.save('Optimal_policies',optpolicies)
    print("Optimal Policies --> 0 - UP; 1 - DOWN; 2 - LEFT; 3 - RIGHT")
    print(optpolicies)
    value_plot(Qval, env)


    env = Maze()
    Q = np.zeros(env.snum,env.anum)

    


    # def policy(state,action):

