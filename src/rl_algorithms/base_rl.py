from numpy import zeros as np_zeros
from blackhc import mdp

class BaseRL(object):
    def __init__(self):
        raise NotImplementedError

    def _get_action_index(self, state_index):
        raise NotImplementedError

    def _update_params(self, t, state_index, action_index, next_state_index, reward):
        raise NotImplementedError

    def learn(self, total_timesteps, store_reward=False, store_estimated_reward=False):
        if store_reward:
            self.reward = np_zeros(total_timesteps)
        if store_estimated_reward:
            from ..infhmdp_utils import expected_reward
            self.estimated_reward = np_zeros(total_timesteps)

        state_index = self.env.reset()
        for t in range(total_timesteps):
            action_index = self._get_action_index(state_index)
            next_state_index, reward, _, _ = self.env.step(action_index)
            self._update_params(t, state_index, action_index, next_state_index, reward)

            if store_reward:
                self.reward[t] = reward
            if store_estimated_reward:
                self.estimated_reward[t] = expected_reward(self.env.mdp.R, self.env.mdp.P, self.policy)

            state_index = next_state_index

    def predict(self, observation: mdp.State):
        assert isinstance(observation, mdp.State)
        action = self.env.mdp.actions[self.policy[observation.index]]
        return action, None
