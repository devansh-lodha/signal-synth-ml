import torch
from torch import nn
import numpy as np

class LinearRegressionModel(nn.Module):
    """
    A simple linear regression model.
    Maps an input of `in_features` dimensions to `out_features` dimensions.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class SignalMLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) for signal reconstruction.
    This model takes RFFs as input and uses hidden layers with non-linear
    activations to learn a complex mapping to the signal values.
    """
    def __init__(self, in_features: int, out_features: int, hidden_features: int, num_hidden_layers: int):
        """
        Initializes the MLP model.

        Args:
            in_features (int): Dimensionality of the input features (RFFs).
            out_features (int): Dimensionality of the output (e.g., 3 for RGB, 1 for audio).
            hidden_features (int): Number of neurons in each hidden layer.
            num_hidden_layers (int): Number of hidden layers.
        """
        super().__init__()
        
        layers = [nn.Linear(in_features, hidden_features), nn.GELU()]
        
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.Linear(hidden_features, hidden_features),
                nn.GELU()
            ])
            
        layers.append(nn.Linear(hidden_features, out_features))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We apply a final tanh activation to keep the output in the [-1, 1] range,
        # which is common for representing normalized signals.
        return torch.tanh(self.net(x))

class SineLayer(nn.Module):
    """
    A sine-based activation layer with a frequency factor 'w0'.
    This is the core building block of a SIREN model.
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, w0=30.0):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.is_first = is_first
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # Special initialization for the first layer
                self.linear.weight.uniform_(-1 / self.linear.in_features, 
                                             1 / self.linear.in_features)
            else:
                # Initialization for subsequent layers
                self.linear.weight.uniform_(-np.sqrt(6 / self.linear.in_features) / self.w0, 
                                             np.sqrt(6 / self.linear.in_features) / self.w0)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))

class SIREN(nn.Module):
    """
    The complete Sinusoidal Representation Network.
    """
    def __init__(self, d_in, d_hidden, d_out, num_layers, w0=30.0):
        super().__init__()
        
        layers = [SineLayer(d_in, d_hidden, is_first=True, w0=w0)]
        for _ in range(num_layers - 1):
            layers.append(SineLayer(d_hidden, d_hidden, w0=w0))
        
        # The final layer is a standard linear layer to map to output values (e.g., RGB)
        final_linear = nn.Linear(d_hidden, d_out)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / d_hidden) / w0, 
                                          np.sqrt(6 / d_hidden) / w0)
        layers.append(final_linear)
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # We process the raw coordinates and apply a final sigmoid to ensure
        # the output is in the [0, 1] range for pixel values.
        x = self.net(x)
        return torch.sigmoid(x)