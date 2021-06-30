import numpy as np
import matplotlib.pyplot as plt 

from rlapse.mdps.mdp import ToyBrokerMDP
from rlapse.algorithms.qlearning import Qlearner
from rlapse.algorithms.rlapse import RLAPSE
from rlapse.utils.infhmdp import ValueIteration, expected_reward

# for reproducibility
SEED = 32
np.random.seed(seed=SEED)

# choose larger T if Qlearner with gamma=0.9 did not converge
T = 700
N_SUPPLIERS = 2
N_PRICES = 2

# generate a controlled ToyBrokerMDP, and validate it
mdp = ToyBrokerMDP(controlled=True)
mdp.validate()

# time step at which to start the orchestrator
T_START = mdp.n_states * mdp.n_states * mdp.n_actions

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
rl = RLAPSE(env, a0, a1, t_start=T_START)

# learn a policy from the environment;
# set store_estimated_reward flag to True to save the estimated reward at each time step
a0.learn(total_timesteps=T, store_estimated_reward=True, verbose=True)
a1.learn(total_timesteps=T, store_estimated_reward=True, verbose=True)
rl.learn(total_timesteps=T, store_estimated_reward=True, verbose=True)

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
