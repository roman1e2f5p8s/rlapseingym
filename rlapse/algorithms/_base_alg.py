import numpy as np
from blackhc import mdp


def _box_state_to_state_index(state, observation_space):
    L = observation_space.low.flatten()
    H = observation_space.high.flatten()
    W = H - L + 1
    O = state.flatten()
    K = len(O)
    #return int(sum([(O[i] - L[i]) * W[i]**(K - 1 - i) for i in range(K)]))
    return int(sum([(O[i] - L[i]) * W[i-1]**(K - 1 - i) for i in range(K)])-1)


def _get_state_index(state, observation_space):
    if np.isscalar(state):
        return state
    else:
        return _box_state_to_state_index(state, observation_space)


class BaseRLalg(object):
    def __init__(self):
        raise NotImplementedError

    def _get_action_index(self, state_index):
        raise NotImplementedError

    def _update_params(self, t, state_index, action_index, next_state_index, reward):
        raise NotImplementedError

    def learn(self, total_timesteps, store_reward=False, store_estimated_reward=False, verbose=False):
        if store_reward:
            self.reward = np.zeros(total_timesteps)

        if store_estimated_reward:
            if hasattr(self.env, 'mdp'):
                from rlapse.mdps.mdp import MDP
                if isinstance(self.env.mdp, MDP):
                    from rlapse.utils.infhmdp import expected_reward
                    self.estimated_reward = np.zeros(total_timesteps)
                else:
                    print('This MDP is not an instanse of \'rlapse.mdps.mdp.MDP\',',
                            'estimated rewards will not be stored.')
                    store_estimated_reward = False
            else:
                print('This Gym environment has no attribute \'mdp\',',
                        'estimated rewards will not be stored.')
                store_estimated_reward = False

        state = self.env.reset()
        state_index = _get_state_index(state, self.env.observation_space)
        for t in range(total_timesteps):
            if verbose:
                print('{}: step #{}'.format(self.__class__.__name__, t), end='\r')

            action_index = self._get_action_index(state_index)
            next_state, reward, _, _ = self.env.step(action_index)
            next_state_index = _get_state_index(next_state, self.env.observation_space)
            self._update_params(t, state_index, action_index, next_state_index, reward)

            if store_reward:
                self.reward[t] = reward

            if store_estimated_reward:
                self.estimated_reward[t] = expected_reward(self.env.mdp.R, self.env.mdp.P, self.policy)

            state_index = next_state_index
        if verbose:
            print('{:8s}: {} steps done.'.format(self.__class__.__name__, total_timesteps))

    '''
    def predict(self, observation: mdp.State):
        assert isinstance(observation, mdp.State)
        action = self.env.mdp.actions[self.policy[observation.index]]
        return action, None
    '''

    def predict(self, state):
        state_index = _get_state_index(state, self.env.observation_space)
        action_index = self.policy[state_index]
        return action_index, None
