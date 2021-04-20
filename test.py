from blackhc.mdp import example, MDPEnv

MDP = example.MULTI_ROUND_NDMP
env = MDPEnv(MDP)
env.__exit__
# env = example.ONE_ROUND_DMDP.to_env()
print(env)
env.reset()
