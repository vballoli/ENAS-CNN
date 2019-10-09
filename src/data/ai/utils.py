import torch 
import torchvision
import os
import logging


def read_model(path):
    """
    Read a completely saved model(not state_dict) from path 
    """
    try:
        model = torch.load(path)
        return model
    except Exception as e:
        raise Exception(e)

def extract_layers(model, layers):
    """
    Recursively extract all layers from the model
    """
    for c in model.children():
        if sum(1 for _ in c.children()) == 0:
            layers.append(c)
        else:
            extract_layers(c, layers)
    return layers

def get_layer_properties(layer, batch_size, input_size):
    M = layer.in_channels
    N = layer.out_channels
    K = layer.kernel_size[0]
    G = layer.groups
    H = input_size[0]
    F = input_size[1]
    B = batch_size
    numerator = B*F*H*M*N*K*K
    denominator = (M*N*K*K)+(B*(M+N)*F*H)
    macs = B*M*N*K*K*F*H/G
    return numerator/denominator, macs