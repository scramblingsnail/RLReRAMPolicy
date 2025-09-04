import torch
import torch.nn as nn
from typing import Union


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


def construct_mlp(dims: [int], activation = None, if_raw_out: bool = True) -> nn.Sequential:
    """
    Construct a MLP (MultiLayer Perceptron)

    dims: the middle dimension, `dims[-1]` is the output dimension of this network
    activation: the activation function
    if_remove_out_layer: if remove the activation function of the output layer.
    """
    if activation is None:
        activation = nn.ReLU
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), activation()])
    if if_raw_out:
        del net_list[-1]  # delete the activation function of the output layer to keep raw output
    return nn.Sequential(*net_list)


# def init_representation_net(config: dict, is_critic: bool):
#     if is_critic:
#         representation_net = CriticRepresentation(action_dim=config['action_dim'],
#                                                   discrete_action=config['discrete_action'],
#                                                   discrete_action_num=config['discrete_action_num'],
#                                                   action_embed_size=config['action_embed_size'],
#                                                   observation_dim=config['observation_dim'],
#                                                   discrete_observation=config['discrete_observation'],
#                                                   discrete_observation_num=config['discrete_observation_num'],
#                                                   state_len=config['state_len'],
#                                                   padding_action=config['padding_action'],
#                                                   observation_embed_size=config['observation_embed_size'])
#     else:
#         representation_net = ActorRepresentation(action_dim=config['action_dim'],
#                                                  discrete_action=config['discrete_action'],
#                                                  discrete_action_num=config['discrete_action_num'],
#                                                  action_embed_size=config['action_embed_size'],
#                                                  observation_dim=config['observation_dim'],
#                                                  discrete_observation=config['discrete_observation'],
#                                                  discrete_observation_num=config['discrete_observation_num'],
#                                                  state_len=config['state_len'],
#                                                  padding_action=config['padding_action'],
#                                                  observation_embed_size=config['observation_embed_size'])
#     return representation_net
#
#
# def init_output_net(config: dict, is_critic: bool, representation: Union[ActorRepresentation, CriticRepresentation]):
#     # output size
#     if config['determinist_action'] or config['discrete_action']:
#         output_size = config['action_dim']
#     else:
#         output_size = config['action_dim'] * 2
#
#     if is_critic:
#         net_name = config['critic_output_net_name']
#         assert isinstance(representation, ActorRepresentation)
#     else:
#         net_name = config['actor_output_net_name']
#         assert isinstance(representation, CriticRepresentation)
#
#     output_net = None
#     if net_name == 'MLP':
#         if is_critic:
#             mlp_neurons = config['critic_mlp_neurons']
#         else:
#             mlp_neurons = config['actor_mlp_neurons']
#         output_net = MLPOutput(input_size=representation.action_embedding_size,
#                                layer_neurons=mlp_neurons,
#                                output_size=output_size)
#     elif net_name == 'LSTM':
#         if is_critic:
#             lstm_neurons = config['critic_lstm_neurons']
#             lstm_layers = config['critic_lstm_layers']
#             mlp_neurons = config['critic_lstm_mlp_neurons']
#         else:
#             lstm_neurons = config['actor_lstm_neurons']
#             lstm_layers = config['actor_lstm_layers']
#             mlp_neurons = config['actor_lstm_mlp_neurons']
#         output_net = LSTMOutput(input_size=representation.action_embedding_size,
#                                 output_size=output_size,
#                                 lstm_neurons=lstm_neurons,
#                                 lstm_layers=lstm_layers,
#                                 mlp_neurons=mlp_neurons)
#     return output_net
#
#
# def init_actor_net(config):
#     representation = init_representation_net(config=config, is_critic=False)
#     output = init_output_net(config=config, is_critic=False, representation=representation)
#     actor = ActorBase(representation_net=representation, output_net=output,
#                       action_dim=config['action_dim'], determinist_action=config['determinist_action'],
#                       discrete_action=config['discrete_action'])
#     return actor
#
#
# def init_critic_net(config):
#     representation = init_representation_net(config=config, is_critic=True)
#     output = init_output_net(config=config, is_critic=True, representation=representation)
#     critic = CriticBase(representation_net=representation, output_net=output)
#     return critic
