from typing import Dict, List, Type, Union

from torch import Tensor, nn


class MultiLayerPerceptron(nn.Module):
    """
    Basic multi-layer perceptron (MLP) neural network. This is a fully connected network with multiple layers.

    [This class was derived heavily from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py#L109]

    :param input_dim: Dimension of the input vector
    :param output_dim: Dimension of the output vectore
    :param net_arch: Architecture of the neural net. Specifically, this represents the number of nodes per layer. The length of this list is the number of layers.
    :param activation_fn: The activation function to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh activation function
    :param with_bias: If set to False, the layers will not learn an additive bias
    :param device: For CUDA, which device to keep network parameters on
    """

    #: Holds neural network layers
    model: nn.Module

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        with_bias: bool = True,
        device: str = "cpu",
    ) -> None:
        super(MultiLayerPerceptron, self).__init__()
        layers: List[nn.Module]
        if len(net_arch) > 0:
            layers = [
                nn.Linear(in_features=input_dim, out_features=net_arch[0], bias=with_bias, device=device),
                activation_fn(),
            ]
        else:
            layers = []

        for idx in range(len(net_arch) - 1):
            layers.append(nn.Linear(in_features=net_arch[idx], out_features=net_arch[idx + 1], bias=with_bias, device=device))
            layers.append(activation_fn())

        if output_dim > 0:
            last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
            layers.append(nn.Linear(in_features=last_layer_dim, out_features=output_dim, bias=with_bias, device=device))
            layers.append(nn.Softmax(dim=0))  # Normalize into a probability distribution
        else:
            raise ValueError("Must specify output dimensions for MLP.")

        self.model = nn.Sequential(*layers)

    def forward(self, observation: Union[Tensor, Dict[str, Tensor]]) -> Tensor:
        """Send an observation through the neural network and return its raw output (should be a probability distribtion)"""
        return self.model(observation)
