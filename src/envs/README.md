# Environments

Making custom environments that are Gymnasium compatible requires some additional overhead. Please see the following documentation

- [Make Your Own Custom Environment](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
- [Environment API](https://gymnasium.farama.org/api/env/#gymnasium.Env.reset)
- [Spaces Datatypes](https://gymnasium.farama.org/api/spaces/fundamental/)

Pay special attention [to Registering Envs](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#registering-envs)

### Updating Environment
If you make changes to the environment, it may be necessary to regenerate the egg with 'make install' 
or (faster) `pip install -e src/envs`

### Debugging Environment
It is recommended to initialize the environment directly instead of using gymnasiums wrapper for debugging. 

For example, replace     

```
env = gymnasium.make("SimpleGrid")
```

with 

```
from envs.simple_gridworld import SimpleGrid

env = SimpleGrid()
