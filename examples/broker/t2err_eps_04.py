import numpy as np
import matplotlib.pyplot as plt 

from rlapse.mdps.mdp import BrokerMDP
from rlapse.algorithms.qlearning import Qlearner
from rlapse.algorithms.rlapse import RLAPSE

# for reproducibility
SEED = 128
np.random.seed(seed=SEED)

# choose larger T if Qlearner with gamma=0.9 did not converge
T = 10000
N_SUPPLIERS = 3
N_PRICES = 3
EPS = 0.4
N_RUNS = 5

t2err = np.empty(shape=(N_RUNS, T))

for run in range(N_RUNS):
    print('Experiment {} out of {}'.format(run+1, N_RUNS))

    # generate a random controlled BrokerMDP, and validate it
    mdp = BrokerMDP(n_suppliers=N_SUPPLIERS, n_prices=N_PRICES, epsilon=EPS)
    mdp.validate()

    # time step at which to start the orchestrator
    t_start = mdp.n_states * mdp.n_states * mdp.n_actions
    
    # convert the MDP to Gym environment
    env = mdp.to_env()
    
    # for reproducibility
    env.observation_space.seed(SEED)
    env.action_space.seed(SEED)
    
    # RL algorithms: Q-learning with gamma 0.0 and 0.9, and RLAPSE
    a0 = Qlearner(env, gamma=0.0)
    a1 = Qlearner(env, gamma=0.9)
    rl = RLAPSE(env, a0, a1, t_start=t_start)
    
    # learn a policy from the environment;
    rl.learn(total_timesteps=T, verbose=True)

    t2err[run, :] = rl.used_a0

# plot the results
fig = plt.figure(figsize=(8, 6))
plt.axhline(y=0, linewidth=0.5, linestyle='-.', color='black')
plt.axvline(x=t_start, linewidth=1, linestyle=':', color='black')
plt.plot(t2err.mean(axis=0), linewidth=2, linestyle='-')
plt.text(x=t_start+(t_start*0.01), y=plt.ylim()[0]+(plt.ylim()[-1]*0.07),
        s=('T0 = {:d}'.format(t_start)), ha='right', va='bottom', rotation=90)
plt.xlabel('Time')
plt.ylabel("Type 2 error")
plt.show()
