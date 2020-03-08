"""
This module defines the composite layer type
"""
from typing import List, Optional

from layer import Layer, Tensor


class Composite(Layer):
    """
    A composite layer consists of multiple sub-layers
    """

    def __init__(
        self, init_strategy: Optional[str] = None, layers: List[Layer] = None,
    ):
        # Add all the layers
        self._sub_layers = []
        if layers is not None:
            for layer in layers:
                self.add_layer(layer)

        input_size = None
        output_size = None
        if self.layers:
            input_size = self.layers[0].shape[0]
            output_size = self.layers[-1].shape[0]

        super(Layer, self).__init__(
            input_size, output_size, None, init_strategy,
        )

        return self

    def add_layer(self, layer: Layer):
        if isinstance(layer, Layer):
            if self.layers:
                output_shape = self.layers[-1].shape[1]
                if layer.shape[0] != output_shape:
                    raise RuntimeError(f"layer should have input shape {output_shape}")

            self._sub_layers.append(layer)
            self._output_size = layer.shape[-1][1]

    @property
    def shape(self):
        return [l.shape for l in self.layers]

    @property
    def layers(self):
        return list(self._sub_layers)

    @property
    def weights(self):
        return [l.weights for l in self.layers]

    def forward(self, X: Tensor, training=False) -> Tensor:
        output = X
        for layer in self.Layers:
            output = layer.forward(output, training=training)
        return output

    def backward(self, dT: Tensor, update=True, learn_rate: float = 0.1) -> Tensor:
        gradient_in = dT
        for layer in self.Layers:
            gradient_in = layer.backward(
                gradient_in, update=update, learn_rate=learn_rate
            )
        return gradient_in
