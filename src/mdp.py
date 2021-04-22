import numpy as np
from blackhc import mdp

from .distribution import Distribution


class MDP(mdp.MDPSpec):
    def __init__(self, n_states: int, n_actions: int, P: np.ndarray, R: np.ndarray):
        assert n_states > 1
        assert n_actions > 1
        assert isinstance(P, np.ndarray)
        assert isinstance(R, np.ndarray)
        assert P.shape == (n_actions, n_states, n_states)
        assert R.shape == (n_states, n_actions)

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
        assert 0 <= epsilon <= 1

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

    def __repr__(self):
        return 'RestaurantMDP(states=%s, actions=%s, state_outcomes=%s, reward_outcomes=%s)' % \
                (self.states, self.actions, dict(self.state_outcomes), dict(self.reward_outcomes))


class RandomMDP(MDP):
    def __init__(self, n_states: int, n_actions: int, controlled: bool, rank1pages: bool,
            P_distribution: Distribution, R_distribution: Distribution):
        assert isinstance(controlled, bool)
        assert isinstance(rank1pages, bool)
        assert isinstance(P_distribution, Distribution)
        assert isinstance(R_distribution, Distribution)

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
