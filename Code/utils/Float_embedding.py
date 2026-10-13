import torch.nn as nn
import torch
from functools import reduce
from operator import mul
import math
import numpy as np


class FloatEmbed(nn.Module):

    def __init__(self, size, drop_p=0.1):
        super(FloatEmbed, self).__init__()
        self.size = size
        self.act = nn.GELU() 
        self.dropout = nn.Dropout(drop_p)
        self.embed = nn.Linear(1, size*2)
        self.embed2 = nn.Linear(size*2, size)
        self.final = nn.LayerNorm(size)

    
    def forward(self, inputs_embeds):
        embeddings = self.embed(inputs_embeds)
        embeddings = self.act(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = self.embed2(embeddings)
        return self.final(embeddings)

