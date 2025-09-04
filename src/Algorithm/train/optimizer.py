import torch
import torch.nn as nn
import torch.optim as optim


def init_optimizer(net: nn.Module, config: dict) -> optim.Optimizer:
    optimizer_name = config['optimizer_name']
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(params=net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(params=net.parameters(), lr=config['lr'], momentum=config['momentum'],
                                  weight_decay=config['weight_decay'])
    else:
        optimizer = optim.SGD(params=net.parameters(), lr=config['lr'], momentum=config['momentum'],
                              weight_decay=config['weight_decay'])
    return optimizer
