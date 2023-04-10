import numpy as np
import joblib


def load_data(embedding_dim):
    word_list, vector_list = [], []
    with open(f"../source/glove.6B.{embedding_dim}d.txt", "r", encoding="utf-8") as f:
        line = f.readline()
        while line != "":
            line = line.strip().split()
            word_list.append(line[0])
            vector_list.append(np.array(line[1:], dtype=np.float))
            line = f.readline()
    vector_list = np.array(vector_list)
    return word_list, vector_list


if __name__ == '__main__':
    words, vectors = load_data(embedding_dim=300)
    joblib.dump(words, f"../source/glove.6B.words.pkl")
    np.save(f"../source/glove.6B.300d.npy", vectors)
