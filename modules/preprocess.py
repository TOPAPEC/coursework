import regex as re
import pandas as pd
import numpy as np
import nltk
from multiprocessing import Pool
from nltk.corpus import stopwords
from tqdm.contrib.concurrent import thread_map
from tqdm import tqdm

nltk.download('stopwords')


def convert(string):
    return eval(string)


cleaning_regex = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
stopwords_eng = stopwords.words('english')


def clean_sentence(sentence):
    for i, word in enumerate(sentence):
        sentence[i] = word.lower()
        sentence[i] = re.sub(cleaning_regex, "", word.lower())
    sentence = [word for word in sentence if not word == "" or word in stopwords_eng]
    return sentence


def clean_further(dataset, cores=12):
    with Pool(processes=cores) as pool:
        return list(tqdm(
            pool.imap(clean_sentence, dataset.loc[:, "text"], chunksize=(dataset.shape[0] // 100)),
            total=dataset.shape[0]))


def parse_lemmatized(file_name, cores=12):
    dataset = pd.read_csv(file_name)
    with Pool(processes=cores) as pool:
        dataset.loc[:, "text"] = pool.map(convert, dataset.loc[:, "text"])
    return dataset


def unite_string(words):
    return " ".join(word for word in words)
