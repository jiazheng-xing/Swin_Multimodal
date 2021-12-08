from typing import ForwardRef
import torch
from torch import nn
from collections import OrderedDict

class text_prompt(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t_x, text):
        o_t_x = t_x[torch.arange(t_x.shape[0]), text.argmax(dim=-1)]
        return o_t_x