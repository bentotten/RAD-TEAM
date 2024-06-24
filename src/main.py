from collections import deque

# import simple_gridworld
# Import the registration file in order to ensure environment registration is triggered
import envs.register_simplegrid  # type: ignore
import numpy as np
import numpy.typing as npt
import torch
from gymnasium import make
from neural_net_cores.simple_mlp import MultiLayerPerceptron as MLP
from torch.distributions import Categorical


def main() -> None:
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init environment
    start: npt.NDArray = np.array([3, 3], dtype=np.float32)
    terminal: npt.NDArray = np.array([0, 0], dtype=np.float32)
    env = make("SimpleGrid-v0", start=start, terminal=terminal, size=5, render_mode="human")
    observation, info = env.reset()

    # Init neural network
    hidden_layers = [16] * 3  # Make 4 layers with 16 neurons each
    observation_space_dim: int = env.observation_space.shape[0] if env.observation_space.shape else 0
    action_space_dim: int = env.action_space.shape[0] if env.action_space.shape else 0
    model = MLP(input_dim=observation_space_dim, output_dim=action_space_dim, net_arch=hidden_layers, device=device)
    model.train()

    # Set up an optimization algorithm for the network
    optimizer = torch.optim.Adam(model.parameters(), lr=0.09)
    optimizer.zero_grad()

    # Save the rewards and probability dists to update the network with
    rewards: list = []
    log_probs: list = []
    steps = 0
    found = 0

    # Train for 100 timesteps
    for i in range(500):
        # Get the neural networks asssesmnet of the situation and then pick an action
        action_prob_dist: torch.Tensor = model.forward(torch.from_numpy(observation))

        # For fun, lets look at the distribution
        print("\n")
        for label, p in enumerate(Categorical(action_prob_dist).probs):  # type: ignore
            print(f"{label}: {100*p:5.2f}%")
        action_prob_dist_cat: Categorical = Categorical(action_prob_dist)  # type: ignore
        action_tensor: torch.Tensor = action_prob_dist_cat.sample()  # type: ignore

        # Convert to an int between 1-4
        action = action_tensor.item() + 1

        # Next step
        observation, reward, done, _, info = env.step(action=action)

        # Save information needed for training
        rewards.append(reward)
        log_probs.append(action_prob_dist_cat.log_prob(action_tensor))  # type: ignore

        steps += 1
        if done:
            found += 1
            print(f"\nFound it in {steps} steps (timestep: {i})")
            print(f"Total reward: {sum(rewards)}!")

            # For simplicity, updating the neural network here
            gamma = 0.99  # Discount factor
            returns: deque = deque()
            # Reverse rewards and apply discount
            R = 0.0
            for reward in reversed(rewards):
                R = float(reward) + (gamma * R)
                returns.appendleft(R)
            returns_tensor = torch.tensor(returns)

            log_prob_tensor = torch.stack(log_probs)

            loss = -log_prob_tensor * returns_tensor
            loss_sum = loss.sum()

            optimizer.zero_grad()
            loss_sum.backward()  # type: ignore
            optimizer.step()

            # Reset
            rewards = []
            log_probs = []
            observation, info = env.reset()
            steps = 0

    env.render()
    # mypy doesnt like the gym wrapper, ignore it
    env.close()  # type: ignore

    print(f"Robot found it {found} times!")


if __name__ == "__main__":
    main()
