import numpy
import torch

import os
import subprocess
from utils.result_utils import *

# computes cosine similarity between two Torch tensors
def get_similarity_scores_tensors(emb1, emb2):
    cos_sim = torch.nn.CosineSimilarity(dim=1)
    val = cos_sim(emb1, emb2)
    return val