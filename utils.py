
import numpy as np
from typing import Iterable

def smooth(rewards: Iterable, window=10):
    return np.convolve(rewards, np.ones(window)/window, mode='valid')
