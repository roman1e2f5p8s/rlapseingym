import numpy as np

from rlapse.utils.distribution import *
from rlapse.utils.infhmdp import greedy_policy, expected_reward, ValueIteration
from rlapse.utils._lrtest import *


def test_distributions_0parg():
    '''
    Test all the numpy distributions that take no positional arguments
    '''
    for dist_name in DISTRIBUTIONS_0PARG:
        np_distribution = getattr(np.random, dist_name)
        d = Distribution(np_distribution)
        d.sample(size=10)


def test_distributions_1parg():
    '''
    Test all the numpy distributions that take 1 positional argument
    '''
    for dist_name in DISTRIBUTIONS_1PARG:
        np_distribution = getattr(np.random, dist_name)
        if not dist_name == 'zipf':
            d = Distribution(np_distribution, [0.5])
        else:
            d = Distribution(np_distribution, [1.5])
        d.sample(size=10)


def test_distributions_2pargs():
    '''
    Test all the numpy distributions that take 2 positional arguments
    '''
    for dist_name in DISTRIBUTIONS_2PARGS:
        np_distribution = getattr(np.random, dist_name)
        if not dist_name == 'multivariate_normal':
            d = Distribution(np_distribution, 1, [1])
        else:
            d = Distribution(np_distribution, [0, 1], np.ones(shape=(2, 2)))
        d.sample(size=10)


def test_distributions_3pargs():
    '''
    Test all the numpy distributions that take 3 positional arguments
    '''
    for dist_name in DISTRIBUTIONS_3PARGS:
        np_distribution = getattr(np.random, dist_name)
        d = Distribution(np_distribution, 0.5, 1, 1)
        d.sample(size=10)


def test_greedy_policy():
    '''
    Test the function that computes the greedy policy
    '''
    R = np.random.uniform(size=(5, 3))
    pi = greedy_policy(R)


def test_expected_reward():
    '''
    Test the function that computes the expected reward
    '''
    n_states, n_actions = 5, 3
    R = np.random.uniform(size=(n_states, n_actions))
    P = np.ones(shape=(n_actions, n_states, n_states)) / n_states
    policy = np.random.randint(low=0, high=n_actions, size=n_states, dtype=int)
    er = expected_reward(R, P, policy)


def test_value_iteration():
    '''
    Test the value iteration algorithm
    '''
    n_states, n_actions = 5, 3
    R = np.random.uniform(size=(n_states, n_actions))
    P = np.ones(shape=(n_actions, n_states, n_states)) / n_states
    vi = ValueIteration(R, P)


def test_lrtest():
    '''
    Test the likelihood-ratio test functions
    '''
    n_states, n_actions = 5, 3
    m = np.random.randint(low=0, high=1000, size=(n_actions, n_states, n_states), dtype=int)
    n = m.sum(axis=2).T   # (n_states, n_actions)
    n_prime = n.sum(axis=1)
    l0 = ln_l0(m, n_prime)
    l1 = ln_l1(m, n)
    L = -2.0 * (l0 - l1)
    DOF = (n_actions - 1) * n_states * (n_states - 1)
    FL = cdf(L, DOF)
