import numpy as np

from src.distribution import Distribution
from src.mdp import RestaurantMDP, RandomMDP, BrokerMDP, ToyBrokerMDP
from src.rl_algorithms.qlearning import Qlearner
from src.rl_algorithms.rlapse import RLAPSE
from src.infhmdp_utils import ValueIteration, expected_reward

SEED = 1
TOTAL_TIMESTEPS = 200
STORE_REWARD = False
STORE_ESTIMATED_REWARD = True
N_STATES = 5
N_ACTIONS = 3
CONTROLLED = True
RANK1PAGES = False

np.random.seed(seed=SEED)

mdp = ToyBrokerMDP(controlled=True)
mdp.validate()
print(mdp.R)
print(mdp.P)
exit()

P_distribution = Distribution(np.random.gamma, shape=1.0, scale=5.0)
R_distribution = Distribution(np.random.gamma, shape=0.1, scale=5.0)
mdp = RandomMDP(
        n_states=N_STATES,
        n_actions=N_ACTIONS,
        controlled=CONTROLLED,
        rank1pages=RANK1PAGES,
        P_distribution=P_distribution,
        R_distribution=R_distribution)
mdp.validate()
# print(mdp.R)
# print(mdp.P)

mdp = RestaurantMDP(epsilon=0.2)
mdp.validate()
print(mdp)
exit()

env = mdp.to_env()
env.reset()
env.observation_space.seed(SEED) # no need do this as it will not be used
env.action_space.seed(SEED)

a0 = Qlearner(env, gamma=0.0)
a1 = Qlearner(env, gamma=0.9)
rlapse = RLAPSE(env, a0, a1)

a0.learn(TOTAL_TIMESTEPS, STORE_REWARD, STORE_ESTIMATED_REWARD)
a1.learn(TOTAL_TIMESTEPS, STORE_REWARD, STORE_ESTIMATED_REWARD)
rlapse.learn(TOTAL_TIMESTEPS, STORE_REWARD, STORE_ESTIMATED_REWARD)

for observation in env.mdp.states:
    action, _ = a1.predict(observation)
    print(observation.index, ':', action.index)

VI = ValueIteration(R=env.mdp.R, P=env.mdp.P)
OPTIMAL_POLICY = VI.policy
OPTIMAL_REWARD = expected_reward(R=env.mdp.R, P=env.mdp.P, policy=OPTIMAL_POLICY)
print('Optimal policy:', OPTIMAL_POLICY)
print('Optimal reward:', OPTIMAL_REWARD)
print('Estimated policy a0:', a0.policy)
print('Estimated reward a0:', a0.estimated_reward[-1])
print('Estimated policy a1:', a1.policy)
print('Estimated reward a1:', a1.estimated_reward[-1])
print('Estimated policy RL:', rlapse.policy)
print('Estimated reward RL:', rlapse.estimated_reward[-1])
print('a0 was used {} times, a1 was used {} times'.format(rlapse.a0_count, rlapse.a1_count))
