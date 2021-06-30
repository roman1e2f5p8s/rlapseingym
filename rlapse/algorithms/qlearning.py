'''
Contains the implementation of the Q-learning algorithm
See "Christopher John Cornish Hellaby Watkins, 1989, Learning from Delayed Rewards.
Ph.D. Dissertation. King’s College, Cambridge, UK,
http://www.cs.rhul.ac.uk/ ̃chrisw/new_thesis.pdf"
and "Eyal Even-Dar and Yishay Mansour. Learning rates for Q-learning.
Journal of machine learning Research,
http://www.jmlr.org/papers/volume5/evendar03a/evendar03a.pdf" for details.

Created on June 10, 2020

@authors: anonymous authors

Contact: anon.email@domain.com
'''

import numpy as np
# from blackhc import mdp
from gym import Env

from rlapse.algorithms._base_alg import BaseRLalg


class Qlearner(BaseRLalg):
    '''
    A class for the Q-learning algorithm
    Required arguments:
        - rand_gen -- random generator, np.random.RandomState
    Optional arguments:
        - state -- initial state, int, 0 <= s < S (default is 0)
        - gamma -- discount factor, float, 0 =< gamma <= 1, default is 0.9
        - epsilon -- percentage of exploration, float, 0 <= epsilon <= 1 (default is 0.2)
        - epsilon_decay_rate -- decay rate of epsilon, float, 0 < epsilon_decay_rate <= 1 (default is 1)
        - epsilon_decay_interval -- interval to decay epsilon , int >= 1 (defaults to 100)
        - omega -- "learning rate", 0.5 < omega < 1 (defaults to 0.7)
    '''
    # def __init__(self, env: mdp.MDPEnv, gamma: float = 0.9, epsilon: float = 0.2,
    def __init__(self, env: Env, gamma: float = 0.9, epsilon: float = 0.2,
            epsilon_decay_rate: float = 1.0, epsilon_decay_interval: int = 100, omega: float = 0.7):
        '''
        Initialization
        '''
        # assert isinstance(env, mdp.MDPEnv)
        assert isinstance(env, Env)
        assert 0.0 <= gamma <= 1.0
        assert 0.0 <= epsilon <= 1.0
        assert 0.0 < epsilon_decay_rate <= 1.0
        assert epsilon_decay_interval >= 1
        assert 0.5 < omega < 1.0

        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_decay_interval = epsilon_decay_interval
        self.omega = omega
        self.iteration = 0
        self.lr = np.ones((env.observation_space.n, env.action_space.n))  # learning rate
        self.Q = np.ones((env.observation_space.n, env.action_space.n))
        self.policy = np.random.randint(0, env.action_space.n, env.observation_space.n, dtype=np.int32)
        self.n = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.int32)

    def _get_action_index(self, state_index):
        '''
        Updates an action for a given state of Q-learner
        Required parameters:
            - rand_gen -- random generator, np.random.RandomState
        Optional parameters:
            - another_alg - another Q-learner which is not used by VSRL at the current step
                (default is None)
        Returns:
            - None
        '''

        if np.random.rand() > self.epsilon: # exploit
            action_index = np.argmax(self.Q[state_index, :])
            self.policy[state_index] = action_index  # update policy
        else:   # explore
            action_index = np.random.choice(self.env.action_space.n)
        
        return action_index

    def _update_params(self, t, state_index, action_index, next_state_index, reward):
        '''
        Updates Q-table and other parameters of the algorithm
        Required parameters:
            - t -- time step, int, t > 0
            - rand_gen -- random generator, np.random.RandomState
            - R_row -- a row from the reward matrix, np.ndarray(A)
            - P_row -- a row from the array of transition probabilities, np.ndarray(S)
        Optional parameters:
            - vsrl -- an object of class for the VSRL algorithm for updating its parameters
                (default is None)
            - another_alg -- another Q-learner which is not used by VSRL at the current step
                (default is None)
        Returns:
            - None
        '''
        
        # decay epsilon
        if not ((self.iteration + 1) % self.epsilon_decay_interval):
            self.epsilon *= self.epsilon_decay_rate

        # update learning rate
        self.n[state_index, action_index] += 1
        self.lr[state_index, action_index] = 1.0 / np.power(1 + self.n[state_index, action_index],
                self.omega)
        
        # update Q-table
        self.Q[state_index, action_index] = (
                self.Q[state_index, action_index] +\
                self.lr[state_index, action_index] * (
                    reward + self.gamma * np.max(self.Q[next_state_index]) -\
                            self.Q[state_index, action_index]
                            )
                )

        self.iteration += 1
