# energy_net/env/register_envs.py

from gymnasium.envs.registration import register

print("Registering EnergyNetEnv-v0")
register(
    id='EnergyNetEnv-v0',
    entry_point='energy_net.env.energy_net_v0:EnergyNetV0',
    # Optional parameters:
    # max_episode_steps=1000,   
    # reward_threshold=100.0,
    # nondeterministic=False,
)
