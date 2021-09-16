import numpy as np
import matplotlib.pyplot as plt 

from rlapse.utils.distribution import Distribution
from rlapse.mdps.mdp import RandomMDP, MDP
from rlapse.algorithms.qlearning import Qlearner
from rlapse.algorithms.rlapse import RLAPSE
from rlapse.utils.infhmdp import ValueIteration, expected_reward
from time import time
import gym


from scipy.sparse import dok_matrix

N = 1000000
a = np.zeros(N, dtype=float)
b = dok_matrix((1, N), dtype=float)
t = time()
for _ in range(100):
    _ = np.argmax(a)
print('np:', (time() - t) / 100)
t = time()
for _ in range(100):
    _ = np.argmax(b.getrow(0))
print('sc:', (time() - t) / 100)
# exit()


class MDPEnv(gym.Env):
    def __init__(self, mdp: MDP):
        self.mdp = mdp

        self._previous_state = None
        self._previous_action = None
        self._state = None
        self._is_done = True
        self.observation_space = gym.spaces.Discrete(self.mdp.n_states)
        self.action_space = gym.spaces.Discrete(self.mdp.n_actions)
        self.start_state = 0

    def reset(self):
        self._previous_state = None
        self._previous_action = None
        self._state = self.start_state
        self._is_done = False
        return self._state

    def step(self, action_index):
        action = action_index
        self._previous_state = self._state
        self._previous_action = action

        if not self._is_done:
            reward = self.mdp.R[self._state, action]

            next_state_probs = self.mdp.P[0, self._state]
            self._state = np.random.choice(self.mdp.n_states, p=next_state_probs)
            self._is_done = False
        else:
            reward = 0

        return self._state, reward, self._is_done, None


# for reproducibility
SEED = 32
np.random.seed(seed=SEED)

# number of states, actions, and time steps
# choose larger T if Qlearner with gamma=0.9 did not converge
S, A, T = 5, 3, 500
# f = np.memmap('data.dat', dtype=np.uint64, mode='w+', shape=(S,S,A))
# del f
'''
from scipy.sparse import dok_matrix
x = np.empty(A, dtype=dok_matrix)
for i in range(len(x)):
    x[i] = dok_matrix((S,S), dtype=np.uint64)
x[0][10, 55] = 1
x[0][10, 55] += 1
x[0][0, 55] = 1
print(x.nbytes)
print(x[0])

exit()
'''

# time step at which to start the orchestrator
T_START = S * S * A
# T_START = 0

# creating distribution wrappers for generation of P and R
P_distribution = Distribution(np.random.gamma, shape=1.0, scale=5.0)
R_distribution = Distribution(np.random.gamma, shape=0.1, scale=5.0)

# generate a random controlled MDP with no rank-1 pages, and validate it
mdp = RandomMDP(
        n_states=S,
        n_actions=A,
        controlled=True,
        rank1pages=False,
        P_distribution=P_distribution,
        R_distribution=R_distribution)
# mdp.validate()

# if necessary, print P and R
# print('P:\n', mdp.P)
# print('R:\n', mdp.R)

# convert the MDP to Gym environment
env = mdp.to_env()
# env = MDPEnv(mdp)

# for reproducibility
env.observation_space.seed(SEED)
env.action_space.seed(SEED)

# RL algorithms: Q-learning with gamma 0.0 and 0.9, and RLAPSE
a0 = Qlearner(env, gamma=0.0)
print('A0 allocated')
a1 = Qlearner(env, gamma=0.9)
print('A1 allocated')
x = time()
rl = RLAPSE(env, a0, a1, t_start=T_START)
print('Algorithms allocated')
print(rl.m.shape)
print(time() - x)
# exit()

# learn a policy from the environment;
# set store_estimated_reward flag to True to save the estimated reward at each time step
# a0.learn(total_timesteps=T, store_estimated_reward=True, verbose=True)
# a1.learn(total_timesteps=T, store_estimated_reward=True, verbose=True)
x = time()
rl.learn(total_timesteps=T, store_estimated_reward=False, verbose=True)
print('Elapsed time:', time() - x)
print(rl.a0_count, rl.a1_count)
exit()

# use the following to predict the action given an observation (i.e. state)
# for observation in range(env.observation_space.n):
    # action, _ = a1.predict(observation)
    # print(observation, ':', action)

# use the value iteration algorithm to compute the optimal policy
VI = ValueIteration(R=env.mdp.R, P=env.mdp.P)
OPTIMAL_POLICY = VI.policy
OPTIMAL_REWARD = expected_reward(R=env.mdp.R, P=env.mdp.P, policy=OPTIMAL_POLICY)

# print the results
print('Expected reward:')
print('\toptimal:         {}'.format(OPTIMAL_REWARD))
print('\testimated by a0: {}'.format(a0.estimated_reward[-1]))
print('\testimated by a1: {}'.format(a1.estimated_reward[-1]))
print('\testimated by rl: {}'.format(rl.estimated_reward[-1]))
print('rl switching counts:')
print('\tto a0: {}'.format(rl.a0_count))
print('\tto a1: {}'.format(rl.a1_count))

# plot the results
fig = plt.figure(figsize=(8, 6))
plt.axvline(x=T_START, linewidth=1, linestyle=':', color='black')
plt.plot(np.ones(T) * OPTIMAL_REWARD, linewidth=4, linestyle='-', color='black', label='Optimal reward')
plt.plot(a0.estimated_reward, linewidth=2, linestyle='-.', label='A0')
plt.plot(a1.estimated_reward, linewidth=2, linestyle='--', label='A1')
plt.plot(rl.estimated_reward, linewidth=2, linestyle='-', label='RLAPSE')
plt.text(x=T_START+(T_START*0.01), y=plt.ylim()[0]+(plt.ylim()[-1]*0.01),
        s=('T0 = {:d}'.format(T_START)), ha='right', va='bottom', rotation=90)
plt.xlabel('Time')
plt.ylabel("Average reward")
plt.legend()
plt.show()
