'''
Contains auxiliary tools for computing policies and rewards for infinite horizon MDPs

Created on June 10, 2020

@authors: anonymous authors

Contact: anon.email@domain.com
'''

import numpy as np
import warnings

# ignore warnings while converting complex numbers like (a+0j) to float
warnings.filterwarnings('ignore', category=np.ComplexWarning)


def greedy_policy(R):
    '''
    Computes the greedy policy for reward R
    Parameters:
        - R -- matrix of reward, np.ndarray((S, A))
    Returns:
        - greedy policy, transposed np.ndarray(S, dtype=int)
    '''
    S = R.shape[0]
    return np.array([R[s].argmax() for s in range(S)], dtype=int).T


def _induced_R(R, policy):
    '''
    Computes the induced reward vector for a given reward distribution R and policy
    Parameters:
        - R -- matrix of reward, np.ndarray((S, A))
        - policy -- policy, np.ndarray(S, dtype=int)
    Returns:
        - induced reward, np.ndarray(S)
    '''
    return np.array([R[s][policy[s]] for s in range(R.shape[0])])


def _induced_MC(P, policy):
    '''
    Computes the induced Markov chain for a given tensor of transition probabilities P and policy
    Parameters:
        - P -- tensor of transition probabilities, np.ndarray((A, S, S))
        - policy -- policy, np.ndarray(S, dtype=int)
    Returns:
        - induced Markov chain (row stochastic matrix), np.ndarray((S, S))
    '''
    return np.array([P[policy[s]][s] for s in range(len(policy))])


def _perron_vec(mc):
    '''
    Computes the Perron vector of Markov chain mc
    Parameters:
        - mc -- Markov chain (row stochastic matrix), np.ndarray((S, S)),
    Returns:
        - Perron vector, np.ndarray(S)
    '''
    eig_val, eig_vec = np.linalg.eig(mc.T)
    perron_root_index = eig_val.argmax()
    try:
        assert np.isclose(eig_val[perron_root_index], 1)
    except AssertionError:
        raise ValueError('Maximum eigenvalue is not equal to 1!')

    return eig_vec[:, perron_root_index] / sum(eig_vec[:, perron_root_index])


def expected_reward(R, P, policy):
    '''
    Computes expected reward for a given reward R, transition probabilities P, and policy
    Parameters:
        - R -- matrix of reward, np.ndarray((S, A))
        - P -- tensor of transition probabilities, np.ndarray((A, S, S)),
        - policy -- policy, np.ndarray(S, dtype=int)
    Returns:
        - expected reward, float
    '''
    return float(np.inner(_perron_vec(_induced_MC(P, policy)), _induced_R(R, policy)))


class ValueIteration:
    '''
    ValueIteration class for computing the optimal policy and its expected return for infinite
        horizon MDPs. See Value Iteration Algorithm on page 161 in "Martin L Puterman, 1994,
        Markov Decision Processes: Discrete Stochastic Dynamic Programming" for details
    Required arguments:
        - R -- reward distribution, np.ndarray((S, A)),
        - P -- transition probabilities, np.ndarray((A, S, S))
        - p0 -- initial distribution, np.ndarray(S)
    Optional arguments:
        - lambda_ -- discount factor, float, 0 <= lambda_ < 1
        - epsilon -- precision, float, epsilon > 0
    '''
    def __init__(self, R, P, p0=None, lambda_=0.9, epsilon=0.001):
        '''
        Initialization
        '''
        self.S, self.A = R.shape
        self.R = R
        self.P = P
        self.p0 = p0 if p0 is not None else np.zeros(self.S) / self.S
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.count = 0  # counter of iterations
        self.STOP = 0.5 * self.epsilon * (1.0 - self.lambda_) / self.lambda_  # stop value
        self.V = np.zeros(self.S)  # the value vector
        self.policy = np.zeros(self.S, dtype=int)
        self.__core__()  # computes the value vector and optimal policy
        self.optimal_return = np.dot(self.p0, self.V)

    def __core__(self):
        '''
        Computes the value vector and optimal policy
        '''
        # auxiliaries
        V_old = np.zeros(self.S)
        V_new = np.zeros(self.S)
        Q = np.zeros(self.A)
        while True:
            for s in range(self.S):
                for a in range(self.A):
                    row = self.P[a, s, :]
                    Q[a] = self.R[s][a] + self.lambda_ * np.dot(row, V_old)
                V_new[s] = np.max(Q)
            if np.linalg.norm(V_new - V_old) < self.STOP:  # stop condition
                self.V = V_new  # update the value vector
                for s in range(self.S):
                    for a in range(self.A):
                        row = self.P[a, s, :]
                        Q[a] = self.R[s][a] + self.lambda_ * np.dot(row, V_new)
                    self.policy[s] = np.argmax(Q)  # update policy
                break
            else:
                for s in range(self.S):
                    V_old[s] = V_new[s]
                self.count += 1
