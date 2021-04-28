import numpy as np

from rlapse.utils.distribution import Distribution
from rlapse.mdps.mdp import RandomMDP
from rlapse.algorithms.qlearning import Qlearner
from rlapse.algorithms.rlapse import RLAPSE


def test_algorithms():
    # number of states, actions, and time steps
    S, A, T = 10, 3, 500
    
    # time step at which to start the orchestrator
    T_START = S * S * A
    
    # creating distribution wrappers for generation of P and R
    P_distribution = Distribution(np.random.uniform)
    R_distribution = Distribution(np.random.uniform)
    
    # generate a random controlled MDP with no rank-1 pages, and validate it
    mdp = RandomMDP(
            n_states=S,
            n_actions=A,
            controlled=True,
            rank1pages=False,
            P_distribution=P_distribution,
            R_distribution=R_distribution)
    mdp.validate()
    
    # convert the MDP to Gym environment
    env = mdp.to_env()
    
    # RL algorithms: Q-learning with gamma 0.0 and 0.9, and RLAPSE
    a0 = Qlearner(env, gamma=0.0)
    a1 = Qlearner(env, gamma=0.9)
    rl = RLAPSE(env, a0, a1, t_start=T_START)
    
    # learn a policy from the environment;
    a0.learn(total_timesteps=T)
    a1.learn(total_timesteps=T)
    rl.learn(total_timesteps=T)
    
    # predict the action given an observation (i.e. state)
    for observation in env.mdp.states:
        action, _ = a0.predict(observation)
        action, _ = a1.predict(observation)
        action, _ = rl.predict(observation)
