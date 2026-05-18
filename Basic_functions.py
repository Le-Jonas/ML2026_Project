import numpy as np
import os
from numba import njit


def convert_to_data(file):
    alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "æ", "ø", "å", " "]
    data = [0] * (len(alphabet) + 1)
    raw = open(file, "r", encoding="utf-8").read().strip()
    N = len(raw)

    total_num = 0
    for char in alphabet:
        num = raw.count(char)
        data[alphabet.index(char)] = num/N
        total_num += num
    data[-1] = (len(raw) - total_num) / N

    return data

def read_files(path):
    data = []
    for file in os.listdir(path):
        data.append(convert_to_data(os.path.join(path, file)))
    return data

