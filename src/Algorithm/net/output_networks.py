import torch
import torch.nn as nn
import numpy as np

from typing import List
from collections import OrderedDict


output_net_names = ('MLP', 'LSTM', 'Transformer')


def construct_mlp(mlp_input_size: int, output_size: int, mlp_neurons: List[int], activation=None):
    mlp = OrderedDict()
    if activation is None:
        activation = nn.ReLU()
    mlp['mlp_input_layer'] = nn.Linear(mlp_input_size, mlp_neurons[0])
    mlp['mlp_input_activation'] = activation
    for idx in range(len(mlp_neurons) - 1):
        mlp['mlp_linear_{}'.format(idx + 1)] = nn.Linear(mlp_neurons[idx], mlp_neurons[idx + 1])
        mlp['mlp_activation_{}'.format(idx + 1)] = activation
    mlp['output_layer'] = nn.Linear(mlp_neurons[-1], output_size)
    mlp['output_activation'] = activation
    return mlp


class MLPOutput(nn.Module):
    def __init__(self, input_size: int, output_size: int, layer_neurons: List[int], activation=None):
        super().__init__()
        # if activation is None:
        #     activation = nn.ReLU()
        # net = OrderedDict()
        # net['input_layer'] = nn.Linear(input_size, layer_neurons[0])
        # net['input_activation'] = activation
        # for idx in range(len(layer_neurons) - 1):
        #     net['linear_{}'.format(idx+1)] = nn.Linear(layer_neurons[idx], layer_neurons[idx+1])
        #     net['activation_{}'.format(idx+1)] = activation
        # net['output_layer'] = nn.Linear(layer_neurons[-1], output_size)
        # net['output_activation'] = activation
        net = construct_mlp(mlp_input_size=input_size, output_size=output_size,
                            mlp_neurons=layer_neurons, activation=activation)
        self.net = nn.ModuleDict(net)

    def forward(self, inputs):
        return self.net(inputs)


class LSTMOutput(nn.Module):
    def __init__(self, input_size, output_size, lstm_neurons: int, lstm_layers: int,
                 mlp_neurons: List[int], activation=None):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_neurons, num_layers=lstm_layers, batch_first=True)
        mlp = construct_mlp(mlp_input_size=lstm_neurons * lstm_layers, output_size=output_size,
                            mlp_neurons=mlp_neurons, activation=activation)
        self.mlp = nn.ModuleDict(mlp)

    def forward(self, inputs):
        r"""
        (batch_size, state_len, input_size)
        return:
            (batch_size, output_size)
        """
        _, state_n = self.lstm(inputs)
        h_n, c_n = state_n
        # (num_layers, batch_size, hidden_num)
        h_n = torch.swapaxes(h_n, 0, 1)
        h_n = h_n.reshape(inputs.size()[0], -1)
        return self.mlp(h_n)


class FETNet(nn.Module):
    r"""
    Priori knowledge:
        For NMOSFET:
        The relationship between saturation current Ids and gate voltage Vg:
        Ids = 1/2 * k * (Vgs - Vth)**2 * (1 + lambda * Vds) - 1/2 * k * lambda * (Vgs - Vth)**3
        Ids = 1/2 * k * (Vgs - Vth)**2 * (-1 + lambda * Vds) + 1/2 * k * lambda * (Vgs - Vth)**3  (- Vds > Vgs - Vth)

        --> Vgs = sqrt(2*abs(Ids) / k) + Vth = sqrt(k_eff * abs(Ids)) + Vth
    """
    def __init__(self, init_k: torch.Tensor, init_vth: torch.Tensor):
        super().__init__()
        self.k_eff = nn.Parameter(init_k)
        self.v_th = nn.Parameter(init_vth)

    def forward(self, Ids):
        v_gs = torch.sqrt(self.k_eff * torch.abs(Ids)) + self.v_th
        return v_gs


# input_size = 3
# lstm_neurons = 5
# lstm_layers = 2
# batch_size = 2
# length = 10
#
# aa = nn.LSTM(input_size=input_size, hidden_size=lstm_neurons, num_layers=lstm_layers, batch_first=True)
# inputs = torch.rand((batch_size, length, input_size))
# print(inputs.size())
# outputs, final = aa(inputs)
# hn = final[0]
# print(outputs.size())
# print(hn.shape)
# print(hn)
# hn = torch.swapaxes(hn, 0, 1)
# hn = hn.reshape(batch_size, -1)
# print(hn)

