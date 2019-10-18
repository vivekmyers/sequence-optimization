from gym.envs.registration import register

register(
    id='batgirl-v0',
    entry_point='gym_batgirl.envs.batgirl_env:Batgirl',
)
