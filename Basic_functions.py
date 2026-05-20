import numpy as np
import os
from numba import njit

chars_set = set(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "æ", "ø", "å", " "])


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

def remove_symbols(text):
    text = text.replace("\n", " ")
    return "".join(ch for ch in text.lower() if ch in chars_set)

def convert_to_data(raw, char_dict):
    data = [0] * len(char_dict)
    N = len(raw)

    total_num = 0
    for char in char_dict:
        num = raw.count(char)
        data[char_dict.index(char)] = num
        total_num += num
    
    for char in char_dict:
        data[char_dict.index(char)] /= total_num

    return data

def extract_header(raw):
    idxs = np.array([i for i, ch in enumerate(raw) if ch == "\n"])
    diff = np.diff(idxs)
    try:
        idx = np.where((diff[:-1] == 1) & (diff[1:] == 1))[0][0]
    except IndexError:
        print(np.where((diff[:-1] == 1) & (diff[1:] == 1)))
        raise IndexError("File {} does not have the expected format.".format(file))
    return raw[idxs[idx]+3:], raw[:idxs[idx]]

def read_files(path, char_dict):
    data = []
    label = []
    for file in os.listdir(path):
        raw = open(os.path.join(path, file), "r", encoding="utf-8").read()
        text, header = extract_header(raw)
        text = text.strip().lower()
        header_lines = header.splitlines()
        date = header_lines[1].split(": ")[1]
        data.append(convert_to_data(text, char_dict))
        label.append(date_to_float(date))
    return data, label

def count_words(path):
    word_count = {}
    for file in os.listdir(path):
        raw = open(os.path.join(path, file), "r", encoding="utf-8").read()
        text, header = extract_header(raw)
        text = text.strip().lower()
        words = remove_symbols(text).split()
        for word in words:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
    return word_count
