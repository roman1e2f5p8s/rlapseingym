import numpy as np
import matplotlib.pyplot as plt 

from rlapse.utils.distribution import Distribution
from rlapse.mdps.mdp import RandomMDP
from rlapse.algorithms.qlearning import Qlearner
from rlapse.algorithms.rlapse import RLAPSE

# for reproducibility
SEED = 32
np.random.seed(seed=SEED)

# number of states, actions, and time steps
# choose larger T if Qlearner with gamma=0.9 did not converge
S, A, T = 5, 3, 500
N_RUNS = 10

# time step at which to start the orchestrator
T_START = S * S * A
t2err = np.empty(shape=(N_RUNS, T))

# creating distribution wrappers for generation of P and R
P_distribution = Distribution(np.random.gamma, shape=1.0, scale=5.0)
R_distribution = Distribution(np.random.gamma, shape=0.1, scale=5.0)

for run in range(N_RUNS):
    print('Experiment {} out of {}'.format(run+1, N_RUNS))

    # generate a random controlled MDP with rank-1 pages, and validate it
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
    
    # for reproducibility
    env.observation_space.seed(SEED)
    env.action_space.seed(SEED)
    
    # RL algorithms: Q-learning with gamma 0.0 and 0.9, and RLAPSE
    a0 = Qlearner(env, gamma=0.0)
    a1 = Qlearner(env, gamma=0.9)
    rl = RLAPSE(env, a0, a1, t_start=T_START)
    
    # learn a policy from the environment;
    rl.learn(total_timesteps=T, verbose=True)

    t2err[run, :] = rl.used_a0

# plot the results
fig = plt.figure(figsize=(8, 6))
plt.axhline(y=0, linewidth=0.5, linestyle='-.', color='black')
plt.axvline(x=T_START, linewidth=1, linestyle=':', color='black')
plt.plot(t2err.mean(axis=0), linewidth=2, linestyle='-')
plt.text(x=T_START+(T_START*0.01), y=plt.ylim()[0]+(plt.ylim()[-1]*0.07),
        s=('T0 = {:d}'.format(T_START)), ha='right', va='bottom', rotation=90)
plt.xlabel('Time')
plt.ylabel("Type 2 error")
plt.show()
