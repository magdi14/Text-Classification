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
