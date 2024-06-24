from gymnasium.envs.registration import register, registry
from simple_gridworld import SimpleGrid

env_id: str = SimpleGrid().env_id

if env_id in registry:
    print(f"Environment '{env_id}' exists, deleting it from the registry")
    del registry[env_id]

print(f"Registering Environment: {env_id}")
register(
    id=env_id,
    entry_point="envs.simple_gridworld:SimpleGrid",
)
