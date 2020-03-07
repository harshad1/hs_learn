"""
This module defines the abstract layer base class.
"""
from typing import Optional, NewType

import numpy as np

# Types
Tensor = NewType("Tensor", np.ndarray)
Bool = [True, False]


class Layer:
    """
    Defines the prototype layer
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
        self._input_size = input_size
        self._output_size = output_size
        self._weights = weights

        self.initialize(init_strategy)

        # Last value
        self._train_output = None

    def forward(self, X: Tensor, training: Bool = False) -> Tensor:
        """
        Execute layer
        """
        result = X

        if training:
            self._train_output = result

        return result

    def backward(self, dT: Tensor, update: Bool = True) -> Tensor:
        """
        Execute backward pass, update weights if required
        """
        # Gradient of the identity function wrt it's input is itself
        return dT

    def initialize(self, strategy=Optional[str]) -> None:
        """
        Initialize weights
        """

    @property
    def weights(self):
        """
        Return weights
        """
        return self._weights
