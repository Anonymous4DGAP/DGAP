# import nltk
# nltk.download("stopwords")

from nltk.corpus import stopwords
from collections import Counter
import re
import numpy as np
from tqdm import tqdm
from stanfordcorenlp import StanfordCoreNLP
from config import get_dataset

dataset = get_dataset()

# To not destroy the sentence structure, stop words and rare words are reserved
stop_words = set()
least_freq = 1
# Remove stop words specifically for long text data set 20ng
if dataset == "20ng":
    stop_words = set(stopwords.words('english'))


# func load texts & labels
def load_dataset(dataset_name):
    with open(f"../raw/{dataset_name}.texts.txt", "r", encoding="latin1") as f:
        text_list = f.read().strip().split("\n")
    with open(f"../raw/{dataset_name}.labels.txt", "r") as f:
        label_list = f.read().strip().split("\n")
    return text_list, label_list


def filter_text(text: str, dataset_name):
    text = text.lower()

    # preprocessing for 20ng
    if dataset_name == '20ng':
        text = text.replace('\n', '')
        regexes_to_remove = [r'from:', r're:', r'subject:', r'distribution:', r'organization:',
                             r'lines:', r'writes:', r'reply-to:']  # remove email head
        for r in regexes_to_remove:
            text = re.sub(r, '', text)
        text = re.sub(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", "", text)  # remove email address
        text = re.sub(r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", "", text)  # rm url
        text = re.sub(r"\d+(-\d+)", "", text)  # remove tel
        text = re.sub(r"\(\d+\)", "", text)  # remove tel
        text = re.sub(r"\d+(\.\d+)", "", text)  # remove decimal

    # preprocessing for ohsumed
    if dataset_name == 'ohsumed':
        text = re.sub(r"\(.*?\)", "", text)  # remove the contents in brackets
        text = re.sub(r"\d+(\.\d+)", "", text)  # remove decimal

    if dataset_name == 'AGNews':
        text = re.sub(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", "", text)  # remove email address
        text = re.sub(r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", "", text)  # rm url

    text = re.sub(r"[^A-Za-z0-9(),!?.\'`]", " ", text)
    text = text.replace("'ll ", " will ")
    text = text.replace("'d ", " would ")
    text = text.replace("'m ", " am ")
    text = text.replace("'s ", " is ")
    text = text.replace("'re ", " are ")
    text = text.replace("'ve ", " have ")
    text = text.replace(" can't ", " can not ")
    text = text.replace(" ain't ", " are not ")
    text = text.replace("n't ", " not ")
    text = text.replace(",", " , ")
    text = text.replace(".", " . ")
    text = text.replace("!", " ! ")
    text = text.replace("(", " ( ")
    text = text.replace(")", " ) ")
    text = text.replace("?", " ? ")
    text = re.sub(r"\s{2,}", " ", text)
    word_list = text.strip().split()
    text = " ".join(word_list)

    if dataset_name == '20ng':
        text = re.sub(r"(\W\s)\1+", "\\1", text)  # remove repeated punctuation for 20ng
    return text


if __name__ == '__main__':

    # Two word segmentation methods: one is space segmentation, the other is CoreNLP's API segmentation.
    # The latter can match the syntax parsing result of corenlp, so the second method is used by default.
    word_tokenize = 'corenlp'  # 'split'
    texts, labels = load_dataset(dataset)

    # handle texts
    print('Filtering text...')
    texts_clean = []
    for t in tqdm(texts, ascii=True):
        texts_clean.append(filter_text(t, dataset))  # segmentation, lowercase, standardized

    print('Tokenizing text...')

    if word_tokenize == 'split':  # space segmentation

        word2count = Counter([w for t in texts_clean for w in t.split()])
        word_count = [[w, c] for w, c in word2count.items()
                      if c >= least_freq and w not in stop_words]  # remove stop words & rare words
        word2index = {w: i for i, (w, c) in enumerate(word_count)}
        words_list = [[w for w in t.split() if w in word2index] for t in texts_clean]

        texts_remove = [" ".join(ws) for ws in words_list]

        # labels 2 targets
        label2index = {l: i for i, l in enumerate(sorted(set(labels)))}
        targets = [label2index[l] for l in labels]

        # save
        with open(f"../temp/{dataset}.texts.remove.txt", "w") as f:
            f.write("\n".join(texts_remove))

        np.save(f"../temp/{dataset}.targets.npy", targets)

    elif word_tokenize == 'corenlp':  # CoreNLP's API segmentation

        nlp = StanfordCoreNLP('http://localhost', port=9000)  # Start CoreNLP service please refer to ReadME
        all_words_list = []
        texts_token = []
        for t in tqdm(texts_clean, ascii=True):
            tokens = nlp.word_tokenize(t)
            texts_token.append(tokens)
            for w in tokens:
                all_words_list.append(w)
        word2count = Counter(all_words_list)
        word_count = [[w, c] for w, c in word2count.items()
                      if c >= least_freq and w not in stop_words]  # remove stop words & rare words
        word2index = {w: i for i, (w, c) in enumerate(word_count)}
        words_list = [[w for w in t if w in word2index] for t in texts_token]

        texts_remove = [" ".join(ws) for ws in words_list]

        # labels 2 targets
        label2index = {l: i for i, l in enumerate(sorted(set(labels)))}
        targets = [label2index[l] for l in labels]

        # save
        with open(f"../temp/{dataset}.texts.remove.txt", "w") as f:
            f.write("\n".join(texts_remove))

        np.save(f"../temp/{dataset}.targets.npy", targets)
