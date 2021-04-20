import numpy as np

from src.mdp import RestaurantMDP

SEED = 1
np.random.seed(seed=SEED)

mdp = RestaurantMDP(epsilon=0.2)
mdp.validate()

env = mdp.to_env()
env.reset()
# env.observation_space.seed(SEED) # no need do this as it will not be used
env.action_space.seed(SEED)

# for _ in range(10):
print(env.step(env.action_space.sample()))
