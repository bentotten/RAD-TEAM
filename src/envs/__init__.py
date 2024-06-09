from gymnasium.envs.registration import register

register(
    id="SimpleGrid",
    entry_point="envs.simple_gridworld:SimpleGrid",
)
