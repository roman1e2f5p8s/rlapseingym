import numpy as np
from blackhc import mdp

from rlapse.utils.distribution import Distribution
from rlapse.mdps._broker_base import _BrokerBase


class MDP(mdp.MDPSpec):
    def __init__(self, n_states: int, n_actions: int, P: np.ndarray, R: np.ndarray):

        # error checking ***********************************************************
        try:
            assert n_states > 1
        except AssertionError as e:
            print('You failed to provide the number of states > 1!')
            raise e

        try:
            assert n_actions > 1
        except AssertionError as e:
            print('You failed to provide the number of actions > 1!')
            raise e

        try:
            assert isinstance(P, np.ndarray)
        except AssertionError as e:
            print('Transition probabilities tensor must be a numpy array!')
            raise e

        try:
            assert isinstance(R, np.ndarray)
        except AssertionError as e:
            print('Reward matrix must be a numpy array!')
            raise e

        try:
            assert P.shape == (n_actions, n_states, n_states)
        except AssertionError as e:
            print('Shape of the transition probabilities tensor must be ({}, {}, {})!'.\
                    format(n_actions, n_states, n_states))
            raise e

        try:
            assert R.shape == (n_states, n_actions)
        except AssertionError as e:
            print('Shape of the reward matrix must be ({}, {})!'.format(n_states, n_actions))
            raise e
        # error checking end *******************************************************

        super().__init__()

        self.n_states = n_states
        self.n_actions = n_actions
        self.P = P
        self.R = R
        self._add_states()
        self._add_actions()
        self._add_transitions()
    
    def __repr__(self):
        return 'MDP(states=%s, actions=%s, state_outcomes=%s, reward_outcomes=%s)' % (self.states,
                self.actions, dict(self.state_outcomes), dict(self.reward_outcomes))

    def _add_states(self):
        for s in range(self.n_states):
            self.state()

    def _add_actions(self):
        for a in range(self.n_actions):
            self.action()

    def _add_transitions(self):
        for a in range(self.n_actions):
            action = self.actions[a]
            for s in range(self.n_states):
                state = self.states[s]
                for ns in range(self.n_states):
                    self.transition(state, action, mdp.NextState(
                        state=self.states[ns],
                        weight=self.P[a, s, ns]))
                self.transition(state, action, mdp.Reward(value=self.R[s, a], weight=1.0))


class RestaurantMDP(MDP):
    def __init__(self, epsilon: float):

        # error checking
        try:
            assert 0 <= epsilon <= 1
        except AssertionError as e:
            print('Epsilon parameter in the restaurant example must be in range [0, 1]!')
            raise e

        P = np.array([
                [[epsilon, 1 - epsilon],
                 [epsilon, 1 - epsilon]],
                [[1 - epsilon, epsilon],
                 [1 - epsilon, epsilon]]
            ])
        R = np.array([
                    [15.0, 1.0],
                    [2.0, 1.0]
                ])

        super().__init__(n_states=2, n_actions=2, P=P, R=R)

        self.epsilon = epsilon

        # GR - Good Restaurant
        # BR - Bad Restaurant
        self.states[0].name = 'There_is_no_wait_in_GR'
        self.states[1].name = 'There_is_a_wait_in_GR'
        self.actions[0].name = 'Send_user_to_GR'
        self.actions[1].name = 'Send_user_to_BR'

    def __repr__(self):
        return 'RestaurantMDP(states=%s, actions=%s, state_outcomes=%s, reward_outcomes=%s)' % \
                (self.states, self.actions, dict(self.state_outcomes), dict(self.reward_outcomes))


class RandomMDP(MDP):
    def __init__(self, n_states: int, n_actions: int, controlled: bool, rank1pages: bool,
            P_distribution: Distribution, R_distribution: Distribution):

        # error checking ***********************************************************
        try:
            assert isinstance(controlled, bool)
        except AssertionError as e:
            print('Value of \"controlled\" parameter must be of type boolean!')
            raise e

        try:
            assert isinstance(rank1pages, bool)
        except AssertionError as e:
            print('Value of \"rank1pages\" parameter must be of type boolean!')
            raise e

        try:
            assert isinstance(P_distribution, Distribution)
        except AssertionError as e:
            print('Transition probabilities distribution must be of type {}!'.\
                    format(Distribution))
            raise e

        try:
            assert isinstance(R_distribution, Distribution)
        except AssertionError as e:
            print('Reward distribution must be of type {}!'.format(Distribution))
            raise e
        # error checking end *******************************************************

        R = R_distribution.sample(size=(n_states, n_actions))

        P = np.zeros(shape=(n_actions, n_states, n_states))
        if controlled:
            if rank1pages:
                prob_distr_repr = 'p(s\'|s,a) = mu(s\'|a)'
                for a in range(n_actions):
                    p = P_distribution.sample(size=n_states)
                    p /= np.sum(p)
                    for s in range(n_states):
                        P[a, :] = p
            else:
                prob_distr_repr = 'p(s\'|s,a) = mu(s\'|s,a)'
                for a in range(n_actions):
                    for s in range(n_states):
                        p = P_distribution.sample(size=n_states)
                        P[a, s] = p / np.sum(p)
        else:
            if rank1pages:
                prob_distr_repr = 'p(s\'|s,a) = mu(s\')'
                p = P_distribution.sample(size=n_states)
                p /= np.sum(p)
                for a in range(n_actions):
                    for s in range(n_states):
                        P[a, :] = p
            else:
                prob_distr_repr = 'p(s\'|s,a) = mu(s\'|s)'
                for s in range(n_states):
                    p = P_distribution.sample(size=n_states)
                    p /= np.sum(p)
                    for a in range(n_actions):
                        P[a, s] = p

        super().__init__(n_states, n_actions, P, R)

        self.is_controlled = controlled
        self.has_rank1pages = rank1pages
        self.P_distribution = P_distribution
        self.R_distribution = R_distribution
        self.prob_distr_repr = prob_distr_repr

    def __repr__(self):
        return 'RandomMDP(states=%s, actions=%s, state_outcomes=%s, reward_outcomes=%s)' % \
                (self.states, self.actions, dict(self.state_outcomes), dict(self.reward_outcomes))


class BrokerMDP(MDP):
    def __init__(self, n_suppliers: int, n_prices: int, epsilon: float):

        # error checking ***********************************************************
        try:
            assert n_suppliers > 1
        except AssertionError as e:
            print('You failed to provide the number of suppliers > 1!')
            raise e

        try:
            assert n_prices > 1
        except AssertionError as e:
            print('You failed to provide the number of price categories > 1!')
            raise e

        try:
            assert 0 <= epsilon < 1
        except AssertionError as e:
            print('Epsilon parameter in the broker example must be in range [0, 1)!')
            raise e
        # error checking end *******************************************************

        broker_base = _BrokerBase(n_suppliers, n_prices, epsilon)

        super().__init__(n_states=broker_base.n_states, n_actions=n_suppliers, P=broker_base.P,
                R=broker_base.R)

        self.n_suppliers = n_suppliers
        self.n_prices = n_prices
        self.epsilon = epsilon
        self.broker_base = broker_base


class ToyBrokerMDP(MDP):
    def __init__(self, controlled: bool):

        # error checking ***********************************************************
        try:
            assert isinstance(controlled, bool)
        except AssertionError as e:
            print('Value of \"controlled\" parameter must be of type boolean!')
            raise e

        broker_base = _BrokerBase(n_suppliers=2, n_prices=2, toy=True, toy_controlled=controlled)

        super().__init__(n_states=broker_base.n_states, n_actions=2, P=broker_base.P, R=broker_base.R)

        self.n_suppliers = 2
        self.n_prices = 2
        self.controlled = controlled
        self.broker_base = broker_base
