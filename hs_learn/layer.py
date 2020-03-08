"""
This module defines the abstract layer base class.
"""
from typing import Optional, NewType

import numpy as np

# Types
Tensor = NewType("Tensor", np.ndarray)


class Layer:
    """
    Defines the prototype layer. The prototype layer implements identity.
    """

    def __init__(
        self,
        input_size: tuple = (),
        output_size: tuple = (),
        weights: Optional[Tensor] = None,
        init_strategy: Optional[str] = None,
    ):
        """
        Initialize the layer
        """
        self._input_size = None
        self._output_size = None
        self._weights = None

        # Last value
        self._train_output = None
        self._update = None

        self.set_shape(input_size, output_size)
        self.set_weights(weights)
        self.initialize(init_strategy)

        return self

    def set_init_strategy(self, init_strategy):
        self._init_strategy = init_strategy

    @property
    def init_strategy(self):
        return self._init_strategy

    def set_shape(self, input_size, output_size):
        self._input_size = input_size
        self._output_size = output_size

    @property
    def shape(self):
        return [(self._input_size, self.output_size)]

    @staticmethod
    def weights_shape(weights):
        if isinstance(weights, Tensor):
            return weights.shape()
        else:
            return [w.shape() for w in weights]

    def set_weights(self, weights):
        if Layer.weights_shape(weights) == self.shape:
            self._weights = weights
        else:
            # TODO: Replace with log message
            raise RuntimeError("Provided weights did not match layer shape")

    @property
    def weights(self):
        """
        Return weights
        """
        return self._weights

    def forward(self, X: Tensor, training=False) -> Tensor:
        """
        Execute layer
        """
        result = X

        if training:
            self._train_output = result

        return result

    def backward(
        self, dT: Tensor, update=True, learn_rate: float = 0.1
    ) -> Tensor:
        """
        Execute backward pass, update weights if required
        """
        # Gradient of the identity function wrt it's input is itself
        self._update = np.zeros((0), dtype=float)

        return dT

    def initialize(self, strategy=Optional[str]) -> None:
        """
        Initialize weights
        """
