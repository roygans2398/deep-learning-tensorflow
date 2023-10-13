import torch
import torch.nn as nn

class CNN(nn.Module):
  def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
    super().__init__()
    self.in_size = in_size
    self.out_classes = out_classes
    self.filters = filters
    self.pool_every = pool_every
    self.hidden_dims = hidden_dims
    self.conv_layers = self._build_conv_layers()
    self.fc_layers = self._build_fc_layers()

  def _build_conv_layers(self):
    in_channels, in_h, in_w, = tuple(self.in_size)
    layers = []

    for i, num_filters_in_layer in enumerate(self.filters):
      layers.append(nn.Conv2d(in_channels, num_filters_in_layer, kernel_size = 3, padding = 1, stride=1))
      layers.append(nn.ReLU())
      in_channels = num_filters_in_layer
      if(i + 1) % self.pool_every == 0:
        layers.append(nn.MaxPool2d(kernel_size = 2, stride=2))
            
    seq = nn.Sequential(*layers)
    return seq

  def _build_fc_layers(self):
    in_channels, in_h, in_w = tuple(self.in_size)
    layers = []
    h_val = in_h // (2**(len(self.filters) // self.pool_every))
    w_val = in_w // (2**(len(self.filters) // self.pool_every))
    num_features = self.filters[-1] * h_val * w_val

    for dim in self.hidden_dims:
      layers.append(nn.Linear(num_features, dim))
      layers.append(nn.ReLU())
      num_features = dim

    layers.append(nn.Linear(num_features, self.out_classes))
    seq = nn.Sequential(*layers)
    return seq

  def forward(self, x):
    out = self.conv_layers(x)
    out = torch.flatten(out, start_dim=1)
    out = self.fc_layers(out)
    return out

model = CNN(in_size=(3, 32, 32), num_classes=10, filters=[32], pool_every=1, hidden_dims=[100])
