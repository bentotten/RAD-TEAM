from gym.envs.registration import register

register(
    id="RadSearch-v1", entry_point="gym_rad_search.envs:RadSearch",
)
