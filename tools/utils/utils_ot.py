import numpy as np
import sys
import torch
import random


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def load_point_files(files):
    out = []
    for file in files:
        out.append(load_point_file(file))
    return np.array(out)


def load_point_file(filename):
    
    pc = np.load(filename)
    pc = np.transpose(pc, (2, 0, 1))
    pc = pc.astype('float32')
    
    return pc
