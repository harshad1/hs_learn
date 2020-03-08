"""
Defines a few of basic layers
"""
import numpy as np

from layer import Layer, Tensor


class Scale(Layer):
    """
    Scales input by constant value
    """

    def __init__(self, scale, size: tuple = ()):

        if not (
            isinstance(scale, float)
            or isinstance(scale, Tensor)
            and (scale.shape() == size)
        ):
            raise RuntimeError("Scale must be scalar or Tensor matching size")

        self._scale = scale
        super(Layer, self).__init__(size, size)

        return self

    def forward(self, X: Tensor, training=False) -> Tensor:
        return X * self._scale

    def backward(
        self, dT: Tensor, update=True, learn_rate: float = 0.1
    ) -> Tensor:
        return dT * self._scale


class Sum(Layer):
    """
    Sums two inputs
    """
    def __init__(self, size: tuple = ()):
        """
        Initialize the layer
        """
        super(Layer, self).__init__(size, size)
        return self

    def forward(self, X1: Tensor, X2: Tensor, training=False) -> Tensor:
        return X1 + X2

    def backward(
        self, dT: Tensor, update=True, learn_rate: float = 0.1
    ) -> Tensor:




