from numpy import array as np_array, ndarray as np_ndarray
from blackhc import mdp


class MDP(mdp.MDPSpec):
    def __init__(self, n_states: int, n_actions: int, P: np_ndarray, R: np_ndarray):
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
        P = np_array([
                [[epsilon, 1 - epsilon],
                 [epsilon, 1 - epsilon]],
                [[1 - epsilon, epsilon],
                 [1 - epsilon, epsilon]]
            ])
        R = np_array([
                    [15.0, 1.0],
                    [2.0, 1.0]
                ])
        super().__init__(n_states=2, n_actions=2, P=P, R=R)
        self.epsilon = epsilon

    def __repr__(self):
        return 'RestaurantMDP(states=%s, actions=%s, state_outcomes=%s, reward_outcomes=%s)' % \
                (self.states, self.actions, dict(self.state_outcomes), dict(self.reward_outcomes))
