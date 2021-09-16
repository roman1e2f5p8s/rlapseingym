'''
Contains the implementation of the VSRL algorithm

Created on June 10, 2020

@authors: anonymous authors

Contact: anon.email@domain.com
'''

import numpy as np
# from blackhc import mdp
from scipy.sparse import dok_matrix
from gym import Env
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
# from copy import deepcopy

from rlapse.algorithms._base_alg import BaseRLalg
from rlapse.algorithms.qlearning import Qlearner
import rlapse.utils._lrtest as lr


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
    # def __init__(self, env: mdp.MDPEnv, a0: Qlearner, a1: Qlearner,
    def __init__(self, env: Env, a0: Qlearner, a1: Qlearner,
            significance_level: float = 0.01, t_start: int = 100):
        '''
        Initialization
        '''
        # assert isinstance(env, mdp.MDPEnv)
        assert isinstance(env, Env)
        assert isinstance(env.action_space, Discrete) or \
                (isinstance(env.action_space, Box) and env.action_space.dtype == int)
        assert isinstance(env.observation_space, Discrete) or \
                (isinstance(env.observation_space, Box) and env.observation_space.dtype == int)
        assert isinstance(a0, Qlearner)
        assert isinstance(a1, Qlearner)
        assert 0.0 < significance_level <= 1.0
        assert t_start >= 0

        self.env = env
        # self.a0 = deepcopy(a0)
        # self.a1 = deepcopy(a1)
        self.a0 = Qlearner(
                env=env,
                gamma=a0.gamma,
                epsilon=a0.epsilon,
                epsilon_decay_rate=a0.epsilon_decay_rate,
                epsilon_decay_interval=a0.epsilon_decay_interval,
                omega=a0.omega,
                )
        self.a1 = Qlearner(
                env=env,
                gamma=a1.gamma,
                epsilon=a1.epsilon,
                epsilon_decay_rate=a1.epsilon_decay_rate,
                epsilon_decay_interval=a1.epsilon_decay_interval,
                omega=a1.omega,
                )
        self.significance_level = significance_level
        self.t_start = t_start

        if isinstance(env.observation_space, Discrete):
            self.n_states = env.observation_space.n
        else:
            self.n_states = np.prod(env.observation_space.high.flatten() - \
                    env.observation_space.low.flatten() + 1) - 1
        if isinstance(env.action_space, Discrete):
            self.n_actions = env.action_space.n
        else:
            self.n_actions = np.prod(env.action_space.high.flatten() - \
                    env.action_space.low.flatten() + 1) - 1

        # self.m_prime = np.zeros((self.n_states, self.n_states), int)
        self.m_prime = dok_matrix((self.n_states, self.n_states), dtype=np.uint64)
        # self.n_prime = np.zeros(self.n_states, int)
        self.n_prime = dok_matrix((1, self.n_states), dtype=np.uint64)

        # self.m = np.zeros((self.n_actions, self.n_states, self.n_states), int)
        self.m = dok_matrix((self.n_actions * self.n_states, self.n_states), dtype=np.uint64)
        # self.n = np.zeros((self.n_states, self.n_actions), int)
        self.n = dok_matrix((self.n_states, self.n_actions), dtype=np.uint64)

        self.ln_l0 = -np.log(self.n_states)
        self.ln_l1 = -np.log(self.n_states)
        self.freedom_degrees = (self.n_actions - 1) * self.n_states *\
                (self.n_states - 1)
        self.use_a1 = False # flag for switching between the basic algorithms
        self.a0_count = 0  # counter of time steps when algorithm a0 is used
        self.a1_count = 0   # counter of time steps when algorithm a1 is used
        self.used_a0 = np.empty(0, dtype=np.uint8) # array to store switches to a0 at given time step
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

    def _ln_l0_diff(self, state_index, next_state_index):
        '''
        ln_l0_sum_s_axis = np.dot(self.m_prime[:, next_state_index], (
            np.log(self.m_prime[:, next_state_index], where=(self.m_prime[:, next_state_index] > 0)) - \
            np.log(self.n_prime, where=(self.n_prime > 0))
            )
        )
        '''
        val = self.n_prime[0, state_index]
        # row = self.m_prime[state_index]
        row = self.m_prime.getrow(state_index).toarray()[0]
        ln_l0_sum_sp_axis = np.dot(row, (np.log(row, where=(row > 0)) - np.log(val, where=(val > 0))))
        '''
        diff = ln_l0_sum_s_axis + ln_l0_sum_sp_axis - \
            self.m_prime[state_index, next_state_index] * (
            np.log(self.m_prime[state_index, next_state_index],
                where=(self.m_prime[state_index, next_state_index] > 0)) - \
            np.log(self.n_prime[state_index], where=(self.n_prime[state_index] > 0))
        )
        '''
        return ln_l0_sum_sp_axis
        # return diff

    def _ln_l1_diff(self, action_index, state_index, next_state_index):
        '''
        ln_l1_sum_a_axis = np.dot(self.m[:, state_index, next_state_index], (
            np.log(self.m[:, state_index, next_state_index],
                where=(self.m[:, state_index, next_state_index] > 0)) - \
            np.log(self.n[state_index], where=(self.n[state_index] > 0))
            )
        )
        ln_l1_sum_s_axis = np.dot(self.m[action_index, :, next_state_index], (
            np.log(self.m[action_index, :, next_state_index],
                where=(self.m[action_index, :, next_state_index] > 0)) - \
            np.log(self.n[:, action_index], where=(self.n[:, action_index] > 0))
            )
        )
        '''
        val = self.n[state_index, action_index]
        # row = self.m[action_index, state_index]
        row = self.m.getrow(action_index * self.n_states + state_index).toarray()[0]
        ln_l1_sum_sp_axis = np.dot(row, (np.log(row, where=(row > 0)) - np.log(val, where=(val > 0))))
        '''
        diff = ln_l1_sum_a_axis + ln_l1_sum_s_axis + ln_l1_sum_sp_axis - \
            2 * self.m[action_index, state_index, next_state_index] * (
            np.log(self.m[action_index, state_index, next_state_index],
                where=(self.m[action_index, state_index, next_state_index] > 0)) - \
            np.log(self.n[state_index, action_index], where=(self.n[state_index, action_index] > 0))
        )
        '''
        return ln_l1_sum_sp_axis
        # return diff

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
    
        # substract corresponding logs before updating the counts
        self.ln_l0 -= self._ln_l0_diff(state_index, next_state_index)
        self.ln_l1 -= self._ln_l1_diff(action_index, state_index, next_state_index)

        # update counts
        # self.n_prime[state_index] += 1
        self.n_prime[0, state_index] += 1
        self.m_prime[state_index, next_state_index] += 1

        self.n[state_index, action_index] += 1
        # self.m[action_index, state_index, next_state_index] += 1
        self.m[action_index * self.n_states + state_index, next_state_index] += 1

        # add corresponding logs after updating the counts
        self.ln_l0 += self._ln_l0_diff(state_index, next_state_index)
        self.ln_l1 += self._ln_l1_diff(action_index, state_index, next_state_index)

        self.a0._update_params(t, state_index, action_index, next_state_index, reward)
        self.a1._update_params(t, state_index, action_index, next_state_index, reward)

        if t + 1 > self.t_start:  # turn the orchestrator on after <self.t_start> time steps
            # L = -2.0 * (ln_l0(self.m, self.n_prime) - ln_l1(self.m, self.n))
            # L = -2.0 * (ln_l0(self.m, self.n_prime, self.n_actions, self.n_actions) - \
                    # ln_l1(self.m, self.n, self.n_actions, self.n_actions))
            L = -2.0 * (self.ln_l0 - self.ln_l1)
            FL = lr.cdf(L, self.freedom_degrees)

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
