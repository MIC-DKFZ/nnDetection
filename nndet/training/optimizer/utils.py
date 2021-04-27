"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Dict, Sequence

import torch
import torch.nn as nn

import nndet.arch.layers.norm as an

NORM_TYPES = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
              nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
              nn.LayerNorm, nn.GroupNorm, nn.SyncBatchNorm, nn.LocalResponseNorm,
              an.GroupNorm,
              ]


def get_params_no_wd_on_norm(model: torch.nn.Module, weight_decay: float):
    """
    Apply weight decay to model but skip normalization layers

    Args:
        model (torch.nn.Module) : module for parameters
        weight_decay (float) : weight decay for other parameters

    Returns:
        dict: dict with params and weight decay

    See Also:
        https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/2
    """
    identify_parameters(model, {"no_wd": NORM_TYPES})

    return [
        {'params': filter(lambda p: not hasattr(p, "no_wd"), model.parameters()), 'weight_decay': weight_decay},
        {'params': filter(lambda p: hasattr(p, "no_wd"), model.parameters()), 'weight_decay': 0.},
    ]


def identify_parameters(model: torch.nn.Module,
                        type_mapping: Dict[str, Sequence],
                        check_param_exist: bool = True):
    """
    Add attribute to searched module types (can be used to filter for specific modules in parameter list)

    Args:
        model: module to add attributes to
        type_mapping: items specify types of modules to search, key specifies name of attribute
        check_param_exist: check if module already has attribute. Can be used to assure that
            attributes are not overwritten, but can lead to wrong results for shared parameters and
            non "primitive" types
    """
    for module in model.modules():
        for _name, _types in type_mapping.items():
            if any([isinstance(module, _type) for _type in _types]):
                for param in module.parameters():
                    if check_param_exist:
                        assert not hasattr(param, _name)
                    setattr(param, _name, True)


def change_output_layer(model: torch.nn.Module, layer_name: str = "fc",
                        output_channels: int = 2, layer_type=torch.nn.Linear,
                        **kwargs) -> None:
    """
    Change layer of module

    Args:
        model (torch.nn.Module): module where layer should be exchanged
        layer_name (str): name of layer to exchange
        output_channels (int): number of new output channels
        layer_type (class): class of new layer
        **kwargs: keyword arguments passed to constructor of new layer
    """
    if not hasattr(model, layer_name):
        raise ValueError(f"Model does not have layer {layer_name}.")

    old_layer = getattr(model, layer_name)
    input_channels = old_layer.in_features

    setattr(model, layer_name,
            layer_type(input_channels, output_channels, **kwargs))


def freeze_layers(model: torch.nn.Module) -> None:
    """
    Freeze layers
    Use something like "Optim([p for p in self.parameters() if p.requires_grad])"
    to be sure.

    Args:
        model(torch.nn.Module): module to freeze
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_layers(model: torch.nn.Module) -> None:
    """
    Unfreeze layers
    Use something like "Optim([p for p in self.parameters() if p.requires_grad])"
    to be sure.

    Args:
        model(torch.nn.Module): module to freeze
    """
    for param in model.parameters():
        param.requires_grad = True
