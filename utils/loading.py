from collections import defaultdict
from glob import glob

import numpy as np


def load_data():
    list_of_negfiles = glob('./corpus/neg/*.txt')
    list_of_posfiles = glob('./corpus/pos/*.txt')
    samples = []
    labels = []
    for i in list_of_negfiles:
        f = open(i)
        samples.append(f.read())
        labels.append(0)
    for i in list_of_posfiles:
        f = open(i)
        samples.append(f.read())
        labels.append(1)
    return np.array(samples), np.array(labels)


def load_glove_embeddings():
    glove = defaultdict(lambda: np.zeros(shape=(50,)))
    with open('glove/glove.6B.50d.txt') as f:
        for line in f:
            if line == '':
                break
            values = line.split()
            glove[values[0]] = np.array(values[1:])
    return glove
