import numpy as np
import os
from numba import njit

chars_set = set(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "æ", "ø", "å", " "])


def int_convert_true(x):
    """
    Checks if a string can be converted to an integer.
    Returns 1 if it can be converted, otherwise returns 0.
    Input:
    x (str): The string to check.
    Output:
    int: 1 if the string can be converted to an integer, otherwise 0.
    """
    try:
        int(x)
        return 1
    except ValueError:
        return 0

def date_to_float(date_str):
    """
    Converts a date string in the format "YYYY-MM-DD" to a float representing the year, with the month and day as fractions of the year.
    If the date string can be converted to an integer, it returns the float value of that integer instead.
    Input:
    date_str (str): The date string to convert.
    Output:
    float: The converted date as a float.
    """
    if int_convert_true(date_str):
        return float(date_str)
    year, month, day = date_str.split("-")
    return float(year) + float(month) / 12 + float(day) / 365

def remove_symbols(text):
    """
    Removes all characters from the input text that are not in the predefined set of characters (chars_set).
    This means it only keeps lowercase letters (a-z), the Danish characters (æ, ø, å), and spaces. It also replaces newlines with spaces.
    Input:
    text (str): The input text to process.
    Output:
    str: The processed text with only the allowed characters.
    """
    text = text.replace("\n", " ")
    return "".join(ch for ch in text.lower() if ch in chars_set)

def convert_to_data(raw, char_dict):
    """
    Converts the input raw text into a list of frequencies for each character in the char_dict. The frequency is calculated as the count of each character in the raw text divided by the total number of characters counted.
    Input:
    raw (str): The input raw text to process.
    char_dict (list): A list of characters for which to calculate the frequencies.
    Output:
    list: A list of frequencies corresponding to each character in char_dict.
    """
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
    """
    Extracts the header and the main text from the input raw text. 
    The header is defined as the part of the text that comes before the first occurrence of two consecutive newlines, and the main text is defined as the part that comes after these two consecutive newlines.
    Input:
    raw (str): The input raw text to process.
    Output:
    tuple: A tuple containing the main text (str) and the header (str).
    """
    idxs = np.array([i for i, ch in enumerate(raw) if ch == "\n"])
    diff = np.diff(idxs)
    try:
        idx = np.where((diff[:-1] == 1) & (diff[1:] == 1))[0][0]
    except IndexError:
        print(np.where((diff[:-1] == 1) & (diff[1:] == 1)))
        raise IndexError("File {} does not have the expected format.".format(file))
    return raw[idxs[idx]+3:], raw[:idxs[idx]]

def read_files(path, char_dict):
    """
    Reads all files in the specified directory, extracts the header and main text from each file, processes the main text to calculate the frequency of each character in char_dict, and extracts the date from the header to convert it to a float. 
    It returns two lists: one containing the processed data for each file and another containing the corresponding labels (dates as floats).
    Input:
    path (str): The path to the directory containing the files to read.
    char_dict (list): A list of characters for which to calculate the frequencies in the main text of each file.
    Output:
    tuple: A tuple containing two lists: the first list contains the processed data for each file, and the second list contains the corresponding labels (dates as floats).
    """
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
    """
    Counts the frequency of each word in all files in the specified directory. 
    It reads each file, extracts the main text, processes it to remove symbols and convert it to lowercase, and then counts the occurrences of each word. 
    The result is a dictionary where the keys are words and the values are their corresponding frequencies.
    Input:
    path (str): The path to the directory containing the files to read.
    Output:
    dict: A dictionary where the keys are words and the values are their corresponding frequencies across all files in the specified directory.
    """
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
