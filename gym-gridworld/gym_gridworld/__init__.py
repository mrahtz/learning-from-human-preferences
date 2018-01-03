from gym.envs.registration import register

register(
    id='GridWorldNoFrameskip-v4',
    entry_point='gym_gridworld.envs:GridWorldEnv'
)
