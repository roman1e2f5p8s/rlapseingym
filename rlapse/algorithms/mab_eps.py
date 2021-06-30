import numpy as np
# from blackhc import mdp
from gym import Env
from gym.spaces.discrete import Discrete
from rlapse.algorithms._base_alg import BaseRLalg


class MAB_EPS(BaseRLalg):
    # def __init__(self, env: mdp.MDPEnv, epsilon: float = 0.2, epsilon_decay_rate: float = 1.0,
    def __init__(self, env: Env, epsilon: float = 0.2, epsilon_decay_rate: float = 1.0,
            epsilon_decay_interval: int = 100):
        # assert isinstance(env, mdp.MDPEnv)
        assert isinstance(env, Env)
        assert isinstance(env.action_space, Discrete)
        assert isinstance(env.observation_space, Discrete)
        assert 0.0 <= epsilon <= 1.0
        assert 0.0 < epsilon_decay_rate <= 1.0
        assert epsilon_decay_interval >= 1

        self.env = env
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_decay_interval = epsilon_decay_interval

        self.iteration = 0
        self.accum_rew = np.ones((1, env.action_space.n))
        self.Q = np.ones((1, env.action_space.n))
        self.n = np.ones((1, env.action_space.n), dtype=np.int32)
        self.policy = np.random.randint(0, env.action_space.n, env.observation_space.n, dtype=np.int32)

    def _get_action_index(self, state_index):
        if np.random.rand() > self.epsilon: # exploit
            action_index = np.argmax(self.Q[0, :])
            self.policy[state_index] = action_index  # update policy
        else:   # explore
            action_index = np.random.choice(self.env.action_space.n)
        
        return action_index

    def _update_params(self, t, state_index, action_index, next_state_index, reward):
        # decay epsilon
        if not ((self.iteration + 1) % self.epsilon_decay_interval):
            self.epsilon *= self.epsilon_decay_rate

        # update
        self.accum_rew[0, action_index] += reward
        self.n[0, action_index] += 1
        self.Q[0, action_index] = self.accum_rew[0, action_index] /\
                self.n[0, action_index]
        
        self.iteration += 1
