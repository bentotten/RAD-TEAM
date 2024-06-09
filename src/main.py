import gymnasium
import numpy as np
import numpy.typing as npt


def main() -> None:
    start: npt.NDArray = np.array([5, 5], dtype=np.int32)
    terminal: npt.NDArray = np.array([0, 0], dtype=np.int32)

    env = gymnasium.make("SimpleGrid", start=start, terminal=terminal, render_mode="human")
    _ = env.reset()

    # Randomly walk for 20 steps
    for _ in range(20):
        action: int = np.random.randint(1, 4)
        observation, reward, done, _, info = env.step(action=action)
        print((observation, reward, done, False, info))

    env.render()
    # mypy doesnt like the gym wrapper, ignore it
    env.close()  # type: ignore


if __name__ == "__main__":
    main()
