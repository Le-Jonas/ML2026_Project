import numpy as np
from scipy.sparse import coo_matrix
import os

import requests
from bs4 import BeautifulSoup
import time
import re
import unicodedata

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

def split_txt(text, length):
    """
    Splits the input text into chunks of the specified length.
    Input:
    text (str): The input text to split.
    length (int): The length of each chunk.
    Output:
    list: A list of strings, each representing a chunk of the input text.
    """
    return [' '.join(text[i:i+length]) for i in range(0, len(text), length)]

def convert_to_data(raw, char_dict, normalize=True):
    """
    Converts the input raw text into a list of frequencies for each character in the char_dict. The frequency is calculated as the count of each character in the raw text divided by the total number of characters counted.
    Input:
    raw (str): The input raw text to process.
    char_dict (list): A list of characters for which to calculate the frequencies.
    Output:
    list: A list of frequencies corresponding to each character in char_dict.
    """
    data = [0] * len(char_dict)
    for i, char in enumerate(char_dict):
        data[i] = raw.count(char)
    if normalize:
        total = sum(data)
        if total != 0:
            data = [x / total for x in data]
    return data

def convert_to_data_sparse(raw, char_dict, row_idx, normalize=True):
    """
    Converts the input raw text into a list of frequencies for each character in the char_dict. The frequency is calculated as the count of each character in the raw text divided by the total number of characters counted.
    Input:
    raw (str): The input raw text to process.
    char_dict (list): A list of characters for which to calculate the frequencies.
    Output:
    list: A list of frequencies corresponding to each character in char_dict.
    list: A list of row indices corresponding to the characters in char_dict that have a non-zero frequency in the raw text.
    list: A list of column indices corresponding to the characters in char_dict that have a non-zero frequency in the raw text.
    """
    row_ = []
    columns_ = []
    values_ = []
    for i, char in enumerate(char_dict):
        num = raw.count(char)
        if num != 0:
            row_.append(row_idx)
            columns_.append(i)
            values_.append(num)
    if normalize:
        total = sum(values_)
        if total != 0:
            values_ = [x / total for x in values_]
    return values_, row_, columns_


def extract_header(raw):
    """
    Extracts the header and the main text from the input raw text. 
    The header is defined as the part of the text that comes before the first occurrence of two consecutive newlines, and the main text is defined as the part that comes after these two consecutive newlines.
    Input:
    raw (str): The input raw text to process.
    file (str): The name of the file being processed.
    Output:
    tuple: A tuple containing the main text (str) and the header (str).
    """
    idxs = np.array([i for i, ch in enumerate(raw) if ch == "\n"])
    diff = np.diff(idxs)
    try:
        idx = np.where(diff[:-1] == 1)[0][0]
    except IndexError:
        return raw[idxs[2]+8:].strip(), raw[:idxs[2]]
        #print(np.where((diff[:-1] == 1) & (diff[1:] == 1)))
        #raise IndexError("File {} does not have the expected format.".format(file))
    return raw[idxs[idx]+2:].strip(), raw[:idxs[idx]]

def read_files(path, char_dict, sparse=False, length = None):
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
    sparse_data = []
    sparse_rows = []
    sparse_cols = []
    row_idx = 0
    for file in os.listdir(path):
        raw = open(os.path.join(path, file), "r", encoding="utf-8").read()
        text, header = extract_header(raw)
        text = text.strip().lower().replace("\n", " ")
        header_lines = header.splitlines()
        date = header_lines[1].split(": ")[1]
        preach = 0
        organization = np.nan
        for line in header_lines:
            if line.split(": ")[-1] == "Prædiken":
                preach = 1
            if line.split(": ")[0] == "Organisationer og bevægelser":
                organization = line.split(": ")[1]
        if length is not None:
            text_words = text.split(" ")
            text_split = split_txt(text_words, length)
            for text_part in text_split:
                if sparse:
                    row = convert_to_data_sparse(text_part, char_dict, row_idx, normalize=False)
                    sparse_data.extend(row[0])
                    sparse_rows.extend(row[1])
                    sparse_cols.extend(row[2])
                else:
                    row = convert_to_data(text_part, char_dict, normalize=False)
                    data.append(row)
                label.append((date_to_float(date), preach, organization))
                row_idx += 1
        else:
            if sparse:
                row = convert_to_data_sparse(text, char_dict, row_idx, normalize=True)
                sparse_data.extend(row[0])
                sparse_rows.extend(row[1])
                sparse_cols.extend(row[2])
            else:
                row = convert_to_data(text, char_dict, normalize=True)
                data.append(row)
            label.append((date_to_float(date), preach, organization))
            row_idx += 1
    if sparse:
        data = coo_matrix((sparse_data, (sparse_rows, sparse_cols)), shape=(row_idx, len(char_dict))).tocsr()
    else:
        data = np.array(data)
    return data, np.array(label)

def count_words_in_directory(path):
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
        text, _ = extract_header(raw)
        word_count = count_words_in_file(word_count, text)
    return word_count

def count_words_in_file(word_count, text):
    """
    Counts the frequency of each word in the given text and updates the provided word_count dictionary with these frequencies.
    It processes the text to remove symbols and convert it to lowercase, then splits it into words and counts the occurrences of each word. The word_count dictionary is updated in-place, where the keys are words and the values are their corresponding frequencies.
    Input:
    word_count (dict): A dictionary where the keys are words and the values are their corresponding frequencies. This dictionary will be updated with the counts from the given text.
    text (str): The input text to process and count words from.
    Output:
    dict: The updated word_count dictionary with the frequencies of words from the given text.
    """
    text = text.strip().lower()
    words = remove_symbols(text).split()
    for word in words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    return word_count

def find_file_word_counts(path):
    """
    Counts the frequency of all words in each file and appends the specific words and counts to a list.
    Input:
    path (str): The path to the directory containing the files to read.
    Output:
    lists: Two lists one of str and one of int, where the first list contains the words and the second list contains the corresponding counts for each file in the specified directory.
    """
    words = []
    counts = []
    for file in os.listdir(path):
        word_count = {}
        raw = open(os.path.join(path, file), "r", encoding="utf-8").read()
        text, _ = extract_header(raw)
        word_count = count_words_in_file(word_count, text)
        words.append(list(word_count.keys()))
        counts.append(list(word_count.values()))
    return words, counts


def polite_get(url, session, min_delay=0.5, max_delay=1.5, max_retries=5):
    time.sleep(np.random.uniform(min_delay, max_delay))
    for attempt in range(max_retries):
        r = session.get(url)
        if r.status_code == 200:
            return r
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            wait = float(ra) if ra else min(2**attempt, 60)
            time.sleep(wait)
            continue
        if 500 <= r.status_code < 600:
            time.sleep(min(2**attempt, 60))
            continue
        r.raise_for_status()
    raise RuntimeError("Max retries exceeded")


def scrape_tale(r):
    soup = BeautifulSoup(r.text, "html.parser")
    title = soup.find("title").text.strip()
    possible_dates = soup.select("time")
    for possible_date in possible_dates:
        if possible_date.has_attr("datetime"):
            date = possible_date["datetime"]
            break
    if "T" in date:
        date = date.split("T")[0]
    err = False

    if soup.select("div.speech-topics") == []:
        topic_names = []
        topic_categories = []
        topic_names.append(soup.select("article")[0].select("a")[0].text.strip())
        topic_categories.append("Article Type")
        try:
            topic_names.append(soup.select("article")[0].select("p")[0].text.strip())
            topic_categories.append("Author")
        except IndexError:
            err = True
            pass

        paragraphs = soup.select("article")[0].select("p")[1:]
        text = ""
        for p in paragraphs:
            for node in p.descendants:
                if node.name is None:
                    text += node.strip()
                    text += "\n"
        
    else:
        topics = soup.select("div.speech-topics")[0].find_all("a")
        topic_categories = []
        topic_names = []
        for topic in topics:
            topic_category = topic.attrs['title'].split("</span>")[0].split(">")[-1].strip()
            topic_categories.append(topic_category)
            topic_names.append(topic.text.strip())

        text_base = soup.find("div", class_="speech-article-content")
        text = ""
        if text_base is None:
            return "Error_text_not_found", date, topic_categories, topic_names, "Text not found"
        for node in text_base.descendants:
            if node.name is None:
                text += node.strip()
                text += "\n"

    return title, date, topic_categories, topic_names, text, err

def sanitize_filename(name, replacement="_"):
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', replacement, name)
    name = re.sub(r"\s+", replacement, name).strip()
    name = name.strip(" .")
    name = re.sub(rf"{re.escape(replacement)}+", replacement, name)
    return name or "untitled"

def save_tale(title, date, topic_categories, topic_names, text, save_dir):
    title = sanitize_filename(title)
    filename = save_dir + "/" + title + ".txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Title: {title}\n")
        f.write(f"Date: {date}\n")
        for i, topic in enumerate(topic_categories):
            f.write(f"{topic}: {topic_names[i]}\n")
        f.write("\n")
        f.write(text)

def main_scrape(base_url, speeches_url, save_dir):
    session = requests.Session()
    session.headers.update({"User-Agent": "KU MachineLearning2026 FinalProject Bot/1.0"})

    for speech_url in speeches_url:
        full_url = base_url + speech_url
        print(f"Processing: {full_url}")
        r = polite_get(full_url, session)
        title, date, topic_categories, topic_names, text, err = scrape_tale(r)
        if err:
            print(f"Error processing {full_url}: Text not found")
            continue
        save_tale(title, date, topic_categories, topic_names, text, save_dir)

    session.close()

def train_val_test_split(data, labels, val_size=0.2, test_size=0.1):
    total_size = len(labels)
    val_count = int(total_size * val_size)
    test_count = int(total_size * test_size)
    train_count = total_size - val_count - test_count
    
    data_train = data[:train_count]
    labels_train = labels[:train_count]
    
    data_val = data[train_count:train_count + val_count]
    labels_val = labels[train_count:train_count + val_count]
    
    data_test = data[train_count + val_count:]
    labels_test = labels[train_count + val_count:]
    
    return data_train, labels_train, data_val, labels_val, data_test, labels_test