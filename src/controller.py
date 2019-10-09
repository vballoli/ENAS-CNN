import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn import (LSTM, RNN, Conv2d, Embedding, Linear, LSTMCell, RNNCell,
                      Sequential)

from logger import get_logger


class Controller(nn.Module):
    """
    RNN-based controller that samples a neural network architecture (Macro Search only). 
    """

    def __init__(self, 
                search_whole_channels, 
                num_layers,
                num_branches,
                out_filters,
                lstm_size,
                lstm_num_layers,
                tanh_constant, 
                temperature,
                skip_target,
                skip_weight,
                log_config):
        super(Controller, self).__init__()
        
        self.logger = get_logger(__name__, config=log_config)
        self.logger.info("Initializing controller %s", str('.'*10))

        self.search_whole_channels = search_whole_channels
        self.num_layers = num_layers
        self.num_branches = num_branches
        self.out_filters = out_filters
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.tanh_constant = tanh_constant
        self.temperature = temperature
        self.skip_target = skip_target
        self.skip_weight = skip_weight

        self._init_params()
        self.logger.info("Finished controller setup %s", str('.'*20))

    def _init_params(self):
        """
        Initializes controller's parameters
        """
        self.logger.info("Initializing parameters %s", str('.'*10))
        self.w_lstm = LSTM(input_size=self.lstm_size,
                            hidden_size=self.lstm_size,
                            num_layers=self.lstm_num_layers)
        self.g_emb = Embedding(1, self.lstm_size)

        if self.search_whole_channels:
            self.w_emb = Embedding(self.num_branches, self.lstm_size)
            self.w_soft = Linear(self.lstm_size, self.num_branches, bias=False)
        else:
            assert False, "Search Whole Channels False is not implemented"
        
        self.w_attn1 = Linear(self.lstm_size, self.lstm_size, bias=False)
        self.w_attn2 = Linear(self.lstm_size, self.lstm_size, bias=False)
        self.v_attn = Linear(self.lstm_size, 1, bias=False)

        self.logger.info("Finished Initializing parameters %s", str('.'*20))
        self._reset_params()
        
    def _reset_params(self):
        """
        Resets the weight parameters of all Embedding, Linear and LSTM hidden units
        """
        self.logger.info("Resetting parameters %s", str('.'*10))
        for module in self.modules():
            if isinstance(module, Linear) or isinstance(module, Embedding):
                nn.init.uniform(module.weight, -0.1, 0.1)

        nn.init.uniform_(self.w_lstm.weight_hh_l0, -0.1, 0.1)
        nn.init.uniform_(self.w_lstm.weight_ih_l0, -0.1, 0.1)
        self.logger.info("Finished Resetting parameters %s", str('.'*20))

    def forward(self):
        """
        Samples blocks from hidden units
        """
        h0 = None                

        anchors = []
        anchors_w1 = []

        arc_seq = {}
        entropys = []
        log_probs = []
        skip_count = []
        skip_penalties = []

        inputs = self.g_emb.weight
        skip_targets = torch.tensor([1.0 - self.skip_target, self.skip_target])

        for layer_id in range(self.num_layers):
            if self.search_whole_channels:
                inputs = inputs.unsqueeze(0)
                output, hn = self.w_lstm(inputs, h0)
                output = output.squeeze(0)
                h0 = hn

                logit = self.w_soft(output)
                if self.temperature:
                    logit /= self.temperature
                if self.tanh_constant:
                    logit = self.tanh_constant * torch.tanh(logit)

                branch_id_dist = Categorical(logits=logit)
                branch_id = branch_id_dist.sample()

                arc_seq[str(layer_id)] = [branch_id]

                log_prob = branch_id_dist.log_prob(branch_id)
                log_probs.append(log_prob.view(-1))
                entropy = branch_id_dist.entropy()
                entropys.append(entropy)

                inputs= self.w_emb(branch_id)
                inputs = inputs.unsqueeze(0)
            else:
                #(TODO) Implement Search whole channels False
                assert False, "Search whole channel False not implemented"

            output, hn = self.w_lstm(inputs, h0)
            output = output.squeeze(0)

            if layer_id > 0:
                query = torch.cat(anchors_w1, dim=0)
                query = torch.tanh(query + self.w_attn2(output))
                query = self.v_attn(query)
                logit = torch.cat([-query, query], dim=1)
                if self.temperature:
                    logit /= self.temperature
                if self.tanh_constant:
                    logit = self.tanh_constant * torch.tanh(logit)

                skip_dist = Categorical(logits=logit)
                skip = skip_dist.sample().view(layer_id)
                
                arc_seq[str(layer_id)].append(skip)

                skip_prob = torch.sigmoid(logit)
                kl = skip_prob * torch.log(skip_prob / skip_targets)
                kl = torch.sum(kl)
                skip_penalties.append(kl)

                log_prob = skip_dist.log_prob(skip)
                log_prob = torch.sum(log_prob)
                log_probs.append(log_prob.view(-1))

                skip = skip.type(torch.float)
                skip = skip.view(1, layer_id)
                skip_count.append(torch.sum(skip))
                inputs - torch.matmul(skip, torch.cat(anchors, dim=0))
                inputs /= (1.0 + torch.sum(skip))
            else:
                inputs = self.g_emb.weight

            anchors.append(output)
            anchors_w1.append(self.w_attn1(output))

        self.sample_arch = arc_seq
        entropys = torch.cat(entropys)
        self.sample_entropy = torch.sum(entropys)
        log_probs = torch.cat(log_probs)
        self.sample_log_probs = torch.sum(log_probs)
        skip_count = torch.starck(skip_penalties)
        self.skip_penalties = torch.mean(skip_penalties)        
