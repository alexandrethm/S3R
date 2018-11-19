import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# Attention
#
#    Attention Score Functions:
#
#        - MLP (Bahdanau et al. 2015)
#        - Bilinear (Luong et al. 2015)
#        - Dot Product (Luong et al. 2015)
#        - Scaled Dot Product (Vaswani et al. 2017)
#
#    Useful links:
#        - http://www.phontron.com/class/nn4nlp2017/assets/slides/nn4nlp-09-attention.pdf
#        - https://medium.com/syncedreview/a-brief-overview-of-attention-mechanism-13c578ba9129
#        - https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html
#        - http://cnyah.com/2017/08/01/attention-variants/


class BahdanauAttention(nn.Module):

    def __init__(self, hidden_size, query_size, memory_size):
        """
        src: https://github.com/chrisbangun/pytorch-seq2seq_with_attention/blob/master/modules/Attention.py
        pdf: https://arxiv.org/abs/1409.0473
        """
        super(BahdanauAttention, self).__init__()

        self.hidden_size = hidden_size
        self.sofmax = nn.Softmax()

        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.memory_layer = nn.Linear(memory_size, hidden_size, bias=False)
        self.alignment_layer = nn.Linear(hidden_size, 1, bias=False)

    def alignment_score(self, query, keys):
        query = self.query_layer(query)
        keys = self.memory_layer(keys)

        extendded_query = query.unsqueeze(1)
        alignment = self.alignment_layer(F.tanh(extendded_query + keys))
        return alignment.squeeze(2)

    def forward(self, query, keys):
        alignment_score = self.alignment_score(query, keys)
        weight = F.softmax(alignment_score)

        context = weight.unsqueeze(2) * keys

        total_context = context.sum(1)

        return total_context, alignment_score


class LuongAttention(torch.nn.Module):

    def __init__(self, method, hidden_size):
        """
        src: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
        pdf: https://arxiv.org/abs/1508.04025
        """
        super(LuongAttention, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        """
        src: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Modules.py
        pdf: https://arxiv.org/abs/1706.03762
        """
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn
