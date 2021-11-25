import regex as re
import pandas as pd
import numpy as np
import nltk
import functools
import gensim
from multiprocessing import Pool
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.contrib.concurrent import thread_map
from tqdm import tqdm
from gensim.models.fasttext import load_facebook_vectors

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


def row_to_embedding(row, embeddings, embeddings_dim, tfidf_dict):
    row = row[1]
    count = 1e-5
    label = row["label"]
    row = row["text"]
    vector_sum = np.zeros(embeddings_dim)
    for word in row:
        if (word in embeddings) and (word in tfidf_dict[label]):
            count += 1
            vector_sum += embeddings[word] * tfidf_dict[label][word]
    return vector_sum / count


def fetch_embeddings_value(row):
    values = row.split(' ');
    vector = np.asarray(values[1:], "float32")
    return values[0], vector


def unpack_sparse(tpl):
    return tpl[0], tpl[1], tpl[2]


def transform_to_tfidf_prod_embeddings(dataset, embeddings, embeddings_dim, tfidf_dict):
    with Pool(processes=14) as pool:
        return np.asarray(
            list(tqdm(pool.imap(
                functools.partial(row_to_embedding, embeddings=embeddings,
                                  embeddings_dim=embeddings_dim, tfidf_dict=tfidf_dict),
                dataset.iterrows(), chunksize=200000), total=dataset.shape[0]))
        )


def get_glove_reddit_embeddings():
    # Number of words - 1623397
    embeddings = {}
    tmp = []
    with io.open("GloVe.Reddit.120B.300D.txt", "r", encoding='utf-8') as file:
        file.readline()
        for line in tqdm(file, total=1623397):
            tmp.append(line)
    with Pool(processes=14) as pool:
        tmp = list(tqdm(pool.imap(fetch_embeddings_value, tmp, chunksize=200000), total=1623397))
    for word, vector in tqdm(tmp):
        embeddings[word] = vector
    del tmp
    return embeddings


def get_word2vec_embeddings():
    return gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)


def get_fasttext_embeddings():
    return load_facebook_vectors('cc.en.300.bin')


def transform_to_tfidf_prod_embeddings_linear(dataset, embeddings, embeddings_dim, tfidf_dict):
    X = np.empty((dataset.shape[0], embeddings_dim), dtype=np.ndarray)
    for i, line in tqdm(enumerate(dataset.iterrows()), total=dataset.shape[0]):
        X[i] = functools.partial(row_to_embedding, embeddings=embeddings,
                                 embeddings_dim=embeddings_dim, tfidf_dict=tfidf_dict)(line)
    return X


def get_features(dataset, cores=12):
    X_reduced = pd.DataFrame()
    X_reduced.loc[:, "label"] = dataset["label"].unique()
    X_reduced["text"] = [[] for i in range(X_reduced.shape[0])]
    for i, label in enumerate(tqdm(X_reduced.loc[:, "label"])):
        titles = dataset[dataset.label == label]["text"].to_list()
        X_reduced.loc[i, "text"].extend(word for title in titles for word in title)
    with Pool(processes=cores) as pool:
        tmp = list(tqdm(pool.imap(unite_string, X_reduced.loc[:, "text"],
                                  chunksize=(X_reduced.shape[0] // 100)),
                        total=X_reduced.shape[0]))
        X_reduced.loc[:, "text"] = tmp
    tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
    tfidf_matrix = tfidfvectorizer.fit_transform(X_reduced.loc[:, "text"])
    feature_names = tfidfvectorizer.vocabulary_
    feature_names = {v: k for k, v in feature_names.items()}
    return tfidf_matrix, feature_names


def get_tfidf_dict(tfidf_matrix, names, dataset):
    cx = tfidf_matrix.tocoo()
    tfidf_dict = {}
    labels = dataset.loc[:, "label"].unique()
    for i, j, v in tqdm(zip(cx.row, cx.col, cx.data)):
        if labels[i] not in tfidf_dict:
            tfidf_dict[labels[i]] = {}
        tfidf_dict[labels[i]][names[j]] = v
    return tfidf_dict
