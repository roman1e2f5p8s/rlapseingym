import numpy as np

from rlapse.utils.distribution import Distribution
from rlapse.mdps.mdp import RandomMDP
from rlapse.algorithms.qlearning import Qlearner
from rlapse.algorithms.rlapse import RLAPSE
from rlapse.utils.infhmdp import ValueIteration, expected_reward

# for reproducibility
SEED = 1
np.random.seed(seed=SEED)

# number of states, actions, and time steps
S, A, T = 10, 3, 5000

# selecting distributions for generation of P and R
P_distribution = Distribution(np.random.gamma, shape=1.0, scale=5.0)
R_distribution = Distribution(np.random.gamma, shape=0.1, scale=5.0)

# generate a random uncontrolled MDP with rank-1 pages, i.e. CMAB, and validate it
mdp = RandomMDP(
        n_states=S,
        n_actions=A,
        controlled=False,
        rank1pages=False,
        P_distribution=P_distribution,
        R_distribution=R_distribution)
mdp.validate()

# if necessary, print P and R
# print('P:\n', mdp.P)
# print('R:\n', mdp.R)

# convert the MDP to Gym environment
env = mdp.to_env()

# for reproducibility
env.observation_space.seed(SEED)
env.action_space.seed(SEED)

# RL algorithms: Q-learning with gamma 0.0 and 0.9, and RLAPSE
a0 = Qlearner(env, gamma=0.0)
a1 = Qlearner(env, gamma=0.9)
rl = RLAPSE(env, a0, a1)

# learn a policy from the environment;
# set store_estimated_reward flag to True to save the estimated reward at each time step
a0.learn(total_timesteps=T, store_estimated_reward=True, verbose=True)
a1.learn(total_timesteps=T, store_estimated_reward=True, verbose=True)
rl.learn(total_timesteps=T, store_estimated_reward=True, verbose=True)

# use the following to predict the action given an observation (i.e. state)
# for observation in env.mdp.states:
    # action, _ = a1.predict(observation)
    # print(observation.index, ':', action.index)

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
