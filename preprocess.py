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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_text(text):

    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)

    return text


def genetate_pairs(data_path):

    file = open(data_path, 'r')
    content = file.read()
    og_pairs = [[normalize_text(s) for s in line.split('\t')] for line in content.split('\n')]
    MAX_len = 40
    pairs = []
    for pair in og_pairs:
        if len(pair) < 2:
            continue
        words_1 = pair[0].split(' ')
        words_2 = pair[1].split(' ')
        if len(words_1) < MAX_len and len(words_2) < MAX_len:
            pairs.append(pair)
    print("No of sentance pairs: ", len(pairs))
    cPickle.dump(pairs, open("pairs.p", "wb"))

    return pairs


class Language:

    def __init__(self, name):

        self.name = name
        self.num_words = 2
        self.word_to_index = {}
        self.index_to_word = {0: "SOS", 1: "EOS"}
        self.word_count = {}

    def add_word(self, word):

        if word not in self.word_to_index:

            self.word_to_index[word] = self.num_words
            self.word_count[word] = 1
            self.index_to_word[self.num_words] = word
            self.num_words += 1

        else:
            self.word_count[word] += 1

eng = Language("eng")
fra = Language("fra")

def genetate_lang(pairs):


    for pair in pairs:

        if len(pair) < 2:
            continue

        for eng_word in pair[0].split(' '):
            eng.add_word(eng_word)
        for fra_word in pair[1].split(' '):
            fra.add_word(fra_word)
    return eng, fra

SOS_token = 0
EOS_token = 1


def tensor_from_sentence(sentence,lang):

    if lang == 'eng':
      indices = [eng.word_to_index[word] for word in sentence.split(' ')]
    elif lang == 'fra':
      indices = [fra.word_to_index[word] for word in sentence.split(' ')]
    indices.append(EOS_token)

    return torch.tensor(indices, dtype=torch.long, device=device).view(-1, 1)

def tensors_from_pair(pair):

    input_tensor = tensor_from_sentence( pair[0],'eng')
    target_tensor = tensor_from_sentence(pair[1],'fra')

    return (input_tensor, target_tensor)
