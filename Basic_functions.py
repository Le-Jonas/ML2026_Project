import numpy as np
import os
from numba import njit

def int_convert_true(x):
    try:
        int(x)
        return 1
    except ValueError:
        return 0

def date_to_float(date_str):
    if int_convert_true(date_str):
        return float(date_str)
    year, month, day = date_str.split("-")
    return float(year) + float(month) / 12 + float(day) / 365

def convert_to_data(raw):
    alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "æ", "ø", "å", " "]
    data = [0] * (len(alphabet) + 1)
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
    label = []
    for file in os.listdir(path):
        raw = open(os.path.join(path, file), "r", encoding="utf-8").read()
        idxs = np.array([i for i, ch in enumerate(raw) if ch == "\n"])
        diff = np.diff(idxs)
        try:
            idx = np.where((diff[:-1] == 1) & (diff[1:] == 1))[0][0]
        except IndexError:
            print(np.where((diff[:-1] == 1) & (diff[1:] == 1)))
            raise IndexError("File {} does not have the expected format.".format(file))
        text = raw[idxs[idx]+3:]
        header = raw[:idxs[idx]]
        header_lines = header.splitlines()
        date = header_lines[1].split(": ")[1]
        data.append(convert_to_data(text.strip()))
        label.append(date_to_float(date))
    return data, label

