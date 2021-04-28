'''
Contains the implementation of the VSRL algorithm

Created on June 10, 2020

@authors: anonymous authors

Contact: anon.email@domain.com
'''

import numpy as np
from blackhc import mdp
from copy import deepcopy

from rlapse.algorithms._base_alg import BaseRLalg
from rlapse.algorithms.qlearning import Qlearner
from rlapse.utils._lrtest import *


class RLAPSE(BaseRLalg):
    '''
    A class for the VSRL algorithm
    Required arguments:
        - S -- number of states, int
        - A -- number of actions, int
        - T -- number of time steps, int
        - cmab_alg -- an object of the LinUCB or Qlearner (myopic) class, algorithm A_0
        - mdp_alg -- an object of the Qlearner (hyperopic) class, algorithm A_1
    Optional arguments:
        - significance_level -- level of significance, float, 0 <= significance_level <= 1
            (default is 0.01)
        - t_start -- time step when the orchestrator will be turned on, int, 0 < t_start <= T
            (default is 150)
        - use_linucb -- use LinUCB as a greedy algorithm or not. If False, Q-learning with gamma=0 is used
            (defaults to False)
        - start_with_complicated -- start the orchestrator with more complicated algorithm, i.e mdp_alg
            (defaults to False)
    '''
    def __init__(self, env: mdp.MDPEnv, a0: Qlearner, a1: Qlearner,
            significance_level: float = 0.01, t_start: int = 100):
        '''
        Initialization
        '''
        assert isinstance(env, mdp.MDPEnv)
        assert isinstance(a0, Qlearner)
        assert isinstance(a1, Qlearner)
        assert 0.0 < significance_level <= 1.0
        assert t_start >= 1

        self.env = env
        self.a0 = deepcopy(a0)
        self.a1 = deepcopy(a1)
        self.significance_level = significance_level
        self.t_start = t_start

        self.m = np.zeros((env.action_space.n, env.observation_space.n, env.observation_space.n), int)
        self.n = np.zeros((env.observation_space.n, env.action_space.n), int)
        self.n_prime = np.zeros(env.observation_space.n, int)
        self.freedom_degrees = (env.action_space.n - 1) * env.observation_space.n *\
                (env.observation_space.n - 1)
        self.use_a1 = False # flag for switching between the basic algorithms
        self.a0_count = 0  # counter of time steps when algorithm a0 is used
        self.a1_count = 0   # counter of time steps when algorithm a1 is used
        self.used_a0 = np.empty(0, int) # array to store the number of switches to a0 at given time step
        self.policy = a0.policy
    
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

        if not self.use_a1:
            action_index = self.a0._get_action_index(state_index)
        else:
            action_index = self.a1._get_action_index(state_index)

        return action_index

    def _update_params(self, t, state_index, action_index, next_state_index, reward):
        '''
        Updates the parameters of VSRL, 
        Parameters:
            - t -- time step, int, t > 0
            - rand_gen -- random generator, np.random.RandomState
            - R -- reward distribution, np.ndarray((S, A))
            - P -- transition probabilities, np.ndarray((A, S, S))
        Returns:
            - None
        '''

        self.n_prime[state_index] += 1
        self.n[state_index, action_index] += 1
        self.m[action_index, state_index, next_state_index] += 1

        self.a0._update_params(t, state_index, action_index, next_state_index, reward)
        self.a1._update_params(t, state_index, action_index, next_state_index, reward)

        if t + 1 > self.t_start:  # turn the orchestrator on after <self.t_start> time steps
            L = -2.0 * (ln_l0(self.m, self.n_prime) - ln_l1(self.m, self.n))
            FL = cdf(L, self.freedom_degrees)

            self.use_a1 = True if (FL >= 1 - self.significance_level) else False

            if not self.use_a1:  # use algorithm a0
                self.used_a0 = np.append(self.used_a0, 1)
                self.policy = self.a0.policy
                self.a0_count += 1
                self.a1.policy[state_index] = self.policy[state_index]
            else:  # use algorithm a1
                self.used_a0 = np.append(self.used_a0, 0)
                self.policy = self.a1.policy
                self.a1_count += 1
                self.a0.policy[state_index] = self.policy[state_index]
        else:  # use algorithm a0
            self.used_a0 = np.append(self.used_a0, 1)
            self.policy = self.a0.policy
            self.a1.policy[state_index] = self.policy[state_index]
