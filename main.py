import gym
import numpy as np
from blackhc import mdp


class MDP_Env(gym.Env):
    def __init__(self, n_states, n_actions, seed=None, reward_range=(0, 1)):
        super().__init__()

        self.n_states = n_states
        self.n_actions = n_actions

        self.observation_space = gym.spaces.Discrete(n_states)
        self.action_space = gym.spaces.Discrete(n_actions)
        self.reward_range = reward_range

        self.seed(seed)
        self.np_random = np.random.RandomState(seed)
        self.init_state = self.observation_space.sample()
        self.state = self.init_state

    def step(self, action):
        state = self.state
        reward = self.np_random.uniform(self.reward_range[0], self.reward_range[1])
        self.state = self.observation_space.sample()
        done = True
        info = {}

        return state, reward, done, info

    def reset(self):
        self.state = self.init_state
        return self.state

    def seed(self, seed=None):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        return [seed]
        


#mdp = MDP_Env(
#       n_states=3,
#       n_actions=2,
#       seed=1,
#       )
#print(mdp.observation_space)
#print(mdp.init_state)
#print(mdp.step(mdp.action_space.sample()))

spec = mdp.MDPSpec()
spec.state()
spec.state()
spec.state()
print(spec.states)
