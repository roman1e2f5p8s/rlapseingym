import numpy as np
from blackhc import mdp

from rlapse.utils.distribution import Distribution
from rlapse.mdps._broker_base import _BrokerBase


class MDP(mdp.MDPSpec):
    def __init__(self, n_states: int, n_actions: int, P: np.ndarray, R: np.ndarray):

        # error checking ***********************************************************
        try:
            assert n_states > 1
        except AssertionError:
            raise ValueError('You failed to provide the number of states > 1!')

        try:
            assert n_actions > 1
        except AssertionError:
            raise ValueError('You failed to provide the number of actions > 1!')

        try:
            assert isinstance(P, np.ndarray)
        except AssertionError:
            raise TypeError('Transition probabilities tensor must be a numpy array!')

        try:
            assert isinstance(R, np.ndarray)
        except AssertionError:
            raise TypeError('Reward matrix must be a numpy array!')

        try:
            assert np.isclose(P.sum(axis=2).all(), 1)
        except AssertionError:
            raise ValueError('Sum of the rows in P is not eqaul to 1!')

        try:
            assert P.shape == (n_actions, n_states, n_states)
        except AssertionError:
            raise ValueError('Shape of the transition probabilities tensor must be ({}, {}, {})!'.\
                    format(n_actions, n_states, n_states))

        try:
            assert R.shape == (n_states, n_actions)
        except AssertionError:
            raise ValueError('Shape of the reward matrix must be ({}, {})!'.\
                    format(n_states, n_actions))
        # error checking end *******************************************************

        super().__init__()

        self.n_states = n_states
        self.n_actions = n_actions
        self.P = P
        self.R = R
        self._add_states()
        self._add_actions()
        self._add_transitions()

    def validate(self):
        print('Validating MDP... ', end='\r')
        super().validate()
        print('Validating MDP... Done!')
    
    def to_env(self):
        print('Converting MDP to GYM environment... ', end='\r')
        env = super().to_env()
        print('Converting MDP to GYM environment... Done!')
        return env
    
    def __repr__(self):
        return 'MDP(states=%s, actions=%s, state_outcomes=%s, reward_outcomes=%s)' % (self.states,
                self.actions, dict(self.state_outcomes), dict(self.reward_outcomes))
    
    def _add_states(self):
        print('Adding {} states to MDP object... '.format(self.n_states), end='\r')
        for s in range(self.n_states):
            self.state()
        print('Adding {} states to MDP object... Done!'.format(self.n_states))

    def _add_actions(self):
        print('Adding {} actions to MDP object... '.format(self.n_actions), end='\r')
        for a in range(self.n_actions):
            self.action()
        print('Adding {} actions to MDP object... Done!'.format(self.n_actions))

    def _add_transitions(self):
        print('Adding {} transitions to MDP object... '.format(self.n_actions*self.n_states**2),
                end='\r')
        for a in range(self.n_actions):
            action = self.actions[a]
            for s in range(self.n_states):
                state = self.states[s]
                for ns in range(self.n_states):
                    self.transition(state, action, mdp.NextState(
                        state=self.states[ns],
                        weight=self.P[a, s, ns]))
                self.transition(state, action, mdp.Reward(value=self.R[s, a], weight=1.0))
        print('Adding {} transitions to MDP object... Done!'.format(self.n_actions*self.n_states**2))


class RestaurantMDP(MDP):
    def __init__(self, epsilon: float):

        # error checking ***********************************************************
        try:
            assert epsilon >= 0
        except AssertionError:
            raise ValueError('Epsilon parameter in the restaurant example cannot be < 0!')

        try:
            assert epsilon <= 1
        except AssertionError:
            raise ValueError('Epsilon parameter in the restaurant example cannot be > 1!')
        # error checking end *******************************************************

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
        except AssertionError:
            raise TypeError('Value of \"controlled\" parameter must be of type boolean!')

        try:
            assert isinstance(rank1pages, bool)
        except AssertionError:
            raise TypeError('Value of \"rank1pages\" parameter must be of type boolean!')

        try:
            assert isinstance(P_distribution, Distribution)
        except AssertionError:
            raise TypeError('Transition probabilities distribution must be of type {}!'.\
                    format(Distribution))

        try:
            assert isinstance(R_distribution, Distribution)
        except AssertionError:
            raise TypeError('Reward distribution must be of type {}!'.format(Distribution))
        # error checking end *******************************************************

        print('Allocating {}x{} reward matrix... '.format(n_states, n_actions), end='\r')
        R = R_distribution.sample(size=(n_states, n_actions)).astype(float)
        print('Allocating {}x{} reward matrix... Done!'.format(n_states, n_actions))
        # if len(R.shape) == 3:
            # R = R.mean(axis=-1)

        print('Allocating {}x{}x{} tensor of transition probabilities... '.format(n_actions,
            n_states, n_states), end='\r')
        P = np.zeros(shape=(n_actions, n_states, n_states), dtype=float)
        # P = np.zeros(shape=(1, n_states, n_states), dtype=float)
        # t = n_actions
        # n_actions = 1
        # self.P = np.memmap('P.dat', shape=(n_actions, n_states, n_states), dtype=float, mode='w+')
        if controlled:
            if rank1pages:
                prob_distr_repr = 'p(s\'|s,a) = mu(s\'|a)'
                for a in range(n_actions):
                    p = P_distribution.sample(size=n_states).astype(float)
                    if len(p.shape) == 2:
                        p = p.mean(axis=-1)
                    p /= np.sum(p)

                    try:
                        assert p.all() >= 0
                    except AssertionError:
                        raise ValueError(
                                'Negative values for probabilities have been generated! ' + 
                                'Please use different parameters or another distribution.')

                    try:
                        assert p.all() <= 1
                    except AssertionError:
                        raise ValueError(
                                'Values > 1 for probabilities have been generated! ' + 
                                'Please use different parameters or another distribution.')

                    try:
                        assert np.isclose(p.sum(), 1)
                    except AssertionError:
                        raise ValueError(
                                'Non-stochastic row has been generated! ' + 
                                'Please use different parameters or another distribution.')

                    for s in range(n_states):
                        P[a, :] = p
            else:
                prob_distr_repr = 'p(s\'|s,a) = mu(s\'|s,a)'
                for a in range(n_actions):
                    for s in range(n_states):
                        p = P_distribution.sample(size=n_states).astype(float)
                        if len(p.shape) == 2:
                            p = p.mean(axis=-1)
                        p /= np.sum(p)

                        try:
                            assert p.all() >= 0
                        except AssertionError:
                            raise ValueError(
                                    'Negative values for probabilities have been generated! ' + 
                                    'Please use different parameters or another distribution.')

                        try:
                            assert p.all() <= 1
                        except AssertionError:
                            raise ValueError(
                                    'Values > 1 for probabilities have been generated! ' + 
                                    'Please use different parameters or another distribution.')

                        try:
                            assert np.isclose(p.sum(), 1)
                        except AssertionError:
                            raise ValueError(
                                    'Non-stochastic row has been generated! ' + 
                                    'Please use different parameters or another distribution.')

                        P[a, s] = p
        else:
            if rank1pages:
                prob_distr_repr = 'p(s\'|s,a) = mu(s\')'
                p = P_distribution.sample(size=n_states).astype(float)
                if len(p.shape) == 2:
                    p = p.mean(axis=-1)
                p /= np.sum(p)

                try:
                    assert p.all() >= 0
                except AssertionError:
                    raise ValueError(
                            'Negative values for probabilities have been generated! ' + 
                            'Please use different parameters or another distribution.')

                try:
                    assert p.all() <= 1
                except AssertionError:
                    raise ValueError(
                            'Values > 1 for probabilities have been generated! ' + 
                            'Please use different parameters or another distribution.')

                try:
                    assert np.isclose(p.sum(), 1)
                except AssertionError:
                    raise ValueError(
                            'Non-stochastic row has been generated! ' + 
                            'Please use different parameters or another distribution.')

                for a in range(n_actions):
                    for s in range(n_states):
                        P[a, :] = p
            else:
                prob_distr_repr = 'p(s\'|s,a) = mu(s\'|s)'
                for s in range(n_states):
                    p = P_distribution.sample(size=n_states).astype(float)
                    if len(p.shape) == 2:
                        p = p.mean(axis=-1)
                    p /= np.sum(p)

                    try:
                        assert p.all() >= 0
                    except AssertionError:
                        raise ValueError(
                                'Negative values for probabilities have been generated! ' + 
                                'Please use different parameters or another distribution.')

                    try:
                        assert p.all() <= 1
                    except AssertionError:
                        raise ValueError(
                                'Values > 1 for probabilities have been generated! ' + 
                                'Please use different parameters or another distribution.')

                    try:
                        assert np.isclose(p.sum(), 1)
                    except AssertionError:
                        raise ValueError(
                                'Non-stochastic row has been generated! ' + 
                                'Please use different parameters or another distribution.')

                    for a in range(n_actions):
                        P[a, s] = p
        print('Allocating {}x{}x{} tensor of transition probabilities... Done!'.format(n_actions,
            n_states, n_states))
        # n_actions = t

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
        except AssertionError:
            raise ValueError('You failed to provide the number of suppliers > 1!')

        try:
            assert n_prices > 1
        except AssertionError:
            raise ValueError('You failed to provide the number of price categories > 1!')

        try:
            assert epsilon >= 0
        except AssertionError:
            raise ValueError('Epsilon parameter in the broker example cannot be < 0!')

        try:
            assert epsilon <= 1
        except AssertionError:
            raise ValueError('Epsilon parameter in the broker example cannot be > 1!')
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
        except AssertionError:
            raise TypeError('Value of \"controlled\" parameter must be of type boolean!')

        broker_base = _BrokerBase(n_suppliers=2, n_prices=2, toy=True, toy_controlled=controlled)

        super().__init__(n_states=broker_base.n_states, n_actions=2, P=broker_base.P, R=broker_base.R)

        self.n_suppliers = 2
        self.n_prices = 2
        self.controlled = controlled
        self.broker_base = broker_base
