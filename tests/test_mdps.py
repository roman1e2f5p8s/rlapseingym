import numpy as np

from rlapse.mdps.mdp import *
from rlapse.utils.distribution import Distribution


def test_mdp():
    '''
    Test MDP class: generates MPDs from specified P and R
    '''
    n_states, n_actions = 10, 3
    P = np.ones(shape=(n_actions, n_states, n_states)) / n_states
    R = np.random.uniform(size=(n_states, n_actions))
    mdp = MDP(n_states=n_states, n_actions=n_actions, P=P, R=R)
    mdp.validate()
    env = mdp.to_env()
    env.render()


def test_restaurant_mdp():
    '''
    Test RestaurantMDP class: generates MPDs for the restaurant example for a given epsilon
    '''
    mdp = RestaurantMDP(epsilon=0.5)
    mdp.validate()
    env = mdp.to_env()
    env.render()


def test_random_mdp():
    '''
    Test RandomMDP class: generates random MPDs from distributions for P and R
    '''
    P_distribution = Distribution(np.random.uniform, low=0.0, high=1.0)
    R_distribution = Distribution(np.random.normal, loc=0.0, scale=0.1)

    mdp = RandomMDP(n_states=10, n_actions=3, controlled=False, rank1pages=True,
            P_distribution=P_distribution, R_distribution=R_distribution)
    mdp.validate()
    env = mdp.to_env()

    mdp = RandomMDP(n_states=10, n_actions=3, controlled=False, rank1pages=False,
            P_distribution=P_distribution, R_distribution=R_distribution)
    mdp.validate()
    env = mdp.to_env()

    mdp = RandomMDP(n_states=10, n_actions=3, controlled=True, rank1pages=True,
            P_distribution=P_distribution, R_distribution=R_distribution)
    mdp.validate()
    env = mdp.to_env()

    mdp = RandomMDP(n_states=10, n_actions=3, controlled=True, rank1pages=False,
            P_distribution=P_distribution, R_distribution=R_distribution)
    mdp.validate()
    env = mdp.to_env()


def test_broker_mdp():
    '''
    Test BrokerMDP class: generates MPDs for the broker example for given number of suppliers and price
    categories
    '''
    mdp = BrokerMDP(n_suppliers=3, n_prices=3, epsilon=0.4)
    mdp.validate()
    env = mdp.to_env()


def test_toy_broker_mdp():
    '''
    Test ToyBrokerMDP class: generates controlled/uncontrolled MPDs for a toy broker example
    '''
    mdp = ToyBrokerMDP(controlled=True)
    mdp.validate()
    env = mdp.to_env()

    mdp = ToyBrokerMDP(controlled=False)
    mdp.validate()
    env = mdp.to_env()
    env.render()
