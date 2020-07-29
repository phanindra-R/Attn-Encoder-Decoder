from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import re
import os
import time
import copy
import unicodedata
import _pickle as cPickle
from encoder import EncoderRNN
from Attn_decoder import AttnDecoderRNN
from eval import evaluate, evaluateRandomly
from preprocess import Language, normalize_text, genetate_pairs, genetate_lang


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_path = 'Data/eng-fra.txt'
pairs = genetate_pairs(data_path)


eng,fra = genetate_lang(pairs)

print("eng vocab size: ", eng.num_words)
print("fra vocab size: ", fra.num_words)


hidden_size = 256
input_lang = 'eng'
output_lang = 'fra'

encoder1 = EncoderRNN(eng.num_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, fra.num_words, dropout_p=0.1).to(device)

from train import trainIters

trainIters(encoder1, attn_decoder1, 75000, print_every=5000)

print("Evaluating randomly")
evaluateRandomly(encoder1, attn_decoder1)

print("model description")
print("encoder model: \n\n", encoder1, '\n')
print("The state dict keys: \n\n", encoder1.state_dict().keys())
print(" ")
print("attn_decoder model: \n\n", attn_decoder1, '\n')
print("The state dict keys: \n\n", attn_decoder1.state_dict().keys())

print("Saving checkpoints")
torch.save(encoder1.state_dict(), 'checkpoint_enc.pth')
files.download('checkpoint_enc.pth')

torch.save(attn_decoder1.state_dict(), 'checkpoint_dec.pth')
files.download('checkpoint_dec.pth')
