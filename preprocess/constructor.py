import os
import joblib
from tqdm import tqdm
import scipy.sparse as sp
from collections import Counter
import numpy as np
import json
from config import get_dataset


dataset = get_dataset()
input_path = os.sep.join(['..', 'data', dataset])
max_text_len = 800
embedding_dim = 300
window_size = 1  # a bi-direc co-occurrence window, containing 1 target word and 2 words adjacent to it - left & right


# normalize
def normalize_adj(adj):
    row_sum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj_normalized


# calculate edge weight & normalize adjacency matrix
def cal_graph_weight(edges, graphs, nrow, ncol):
    edge_count = Counter(edges).items()
    row = [x for (x, y), c in edge_count]
    col = [y for (x, y), c in edge_count]
    weight = [c for (x, y), c in edge_count]

    # normalize adjacency matrix
    adj = sp.csr_matrix((weight, (row, col)), shape=(nrow, ncol))
    adj_normalized = normalize_adj(adj)
    weight_normalized = [adj_normalized[x][y] for (x, y), c in edge_count]

    graphs.append([row, col, weight_normalized])


def pad_seq(seq, pad_len, pad_elem=0):
    if len(seq) > pad_len:
        return seq[:pad_len]
    return seq + [pad_elem] * (pad_len - len(seq))


def construct_graph(parse_list):

    # build graph
    input_word = []  # the input word nodes, assuming the number of words in the sentence is n
    graphs_occ = []  # topology and weight of co-occurrence edges
    graphs_dep = []  # topology and weight of dependency edges
    graphs_mix = []  # topology and weight of mix edges

    # remake word2index
    print("Get word2index:")
    all_words_list = []
    for doc_id in tqdm(range(len(parse_list)), ascii=True):
        word_node_list = parse_list[doc_id]['word_nodes']
        for w in word_node_list:
            all_words_list.append(w)
    word2count = Counter(all_words_list)
    word_count = [[w, c] for w, c in word2count.items()]
    word2index = {w: i for i, (w, c) in enumerate(word_count)}

    print("Make graph:")
    for doc_id in tqdm(range(len(parse_list)), ascii=True):

        dep_edge_list = parse_list[doc_id]['dep_edges']
        word_node_list = parse_list[doc_id]['word_nodes']

        occ_edges, dep_edges = [], []
        mix_edges = []

        word_node_list = word_node_list[:max_text_len]  # word list (limit maximum length)

        for i in range(len(word_node_list)):  # add co-occurrence edges
            target_idx = i
            for j in range(i - window_size, i + window_size + 1):
                if i != j and 0 <= j < len(word_node_list):
                    source_idx = j
                    occ_edges.append((source_idx, target_idx))
            # add co-occurrence node pairs with the distance of 2
            if i + 2 < len(word_node_list):
                source_idx = i + 2
                occ_edges.append((source_idx, target_idx))
                occ_edges.append((target_idx, source_idx))

        for depend_tuple in dep_edge_list:  # add dependency edges
            source_idx = depend_tuple[0]
            target_idx = depend_tuple[2]
            if source_idx >= max_text_len or target_idx >= max_text_len:
                break
            dep_edges.append((source_idx, target_idx))
            dep_edges.append((target_idx, source_idx))

        mix_edges = list(set(occ_edges) | set(dep_edges))  # get union edges

        id_word_list = [word2index[w] for w in word_node_list]  # word2index.get(w, 0)
        input_word.append(id_word_list)

        # calculate co-occurrence edge weight
        cal_graph_weight(occ_edges, graphs_occ, len(word_node_list), len(word_node_list))

        # calculate dependency edge weight
        cal_graph_weight(dep_edges, graphs_dep, len(word_node_list), len(word_node_list))

        # calculate mix edge weight
        cal_graph_weight(mix_edges, graphs_mix, len(word_node_list), len(word_node_list))

    # The number of nodes and edges of each graph is recorded here
    # to preserve the boundary for restoration after padding.
    len_input = [len(e) for e in input_word]
    len_graphs_occ = [len(x) for x, y, c in graphs_occ]
    len_graphs_dep = [len(x) for x, y, c in graphs_dep]
    len_graphs_mix = [len(x) for x, y, c in graphs_mix]

    # padding input
    pad_len_input = max(len_input)  # Maximum number of nodes in text graph
    pad_len_occ_graph = max(len_graphs_occ)  # Maximum number of co-occurrence edges
    pad_len_dep_graph = max(len_graphs_dep)  # Maximum number of dependency edges
    pad_len_mix_graph = max(len_graphs_mix)  # Maximum number of mix edges

    input_word_pad = [pad_seq(e, pad_len_input, len(word2index)) for e in input_word]  # padding to save as numpy-array
    graphs_occ_pad = [[pad_seq(ee, pad_len_occ_graph) for ee in e] for e in graphs_occ]
    graphs_dep_pad = [[pad_seq(ee, pad_len_dep_graph) for ee in e] for e in graphs_dep]
    graphs_mix_pad = [[pad_seq(ee, pad_len_mix_graph) for ee in e] for e in graphs_mix]

    input_word_pad = np.array(input_word_pad)  # input_word_pad.shape: doc_num, max_node_num
    weight_occ_pad = np.array([c for x, y, c in graphs_occ_pad])  # weight_occ_pad.shape: doc_num, max_edge_num
    graphs_occ_pad = np.array([[x, y] for x, y, c in graphs_occ_pad])  # graphs_occ_pad.shape: doc_num, max_edge_num, 2
    weight_dep_pad = np.array([c for x, y, c in graphs_dep_pad])  # weight_dep_pad.shape: doc_num, max_edge_num
    graphs_dep_pad = np.array([[x, y] for x, y, c in graphs_dep_pad])  # graphs_dep_pad.shape: doc_num, max_edge_num, 2
    weight_mix_pad = np.array([c for x, y, c in graphs_mix_pad])  # weight_mix_pad.shape: doc_num, max_edge_num
    graphs_mix_pad = np.array([[x, y] for x, y, c in graphs_mix_pad])  # graphs_mix_pad.shape: doc_num, max_edge_num, 2

    def get_oov():
        oov = np.random.normal(-0.1, 0.1, embedding_dim)
        return oov

    all_vectors = np.load(f"../source/glove.6B.{embedding_dim}d.npy")
    all_words = joblib.load(f"../source/glove.6B.words.pkl")
    all_word2index = {w: i for i, w in enumerate(all_words)}
    word_set = [w for w, i in word2index.items()]
    word2vec = [all_vectors[all_word2index[w]] if w in all_word2index else get_oov() for w in word_set]
    # add an all-zero vector at the end of the vocab as the vector of [PAD]
    word2vec.append(np.zeros(embedding_dim))

    # save
    joblib.dump(len_input, f"../temp/{dataset}.len.inputs.pkl")
    joblib.dump(len_graphs_occ, f"../temp/{dataset}.len.graphs_occ.pkl")
    joblib.dump(len_graphs_dep, f"../temp/{dataset}.len.graphs_dep.pkl")
    joblib.dump(len_graphs_mix, f"../temp/{dataset}.len.graphs_mix.pkl")

    np.save(f"../temp/{dataset}.input_word.npy", input_word_pad)
    np.save(f"../temp/{dataset}.graphs_occ.npy", graphs_occ_pad)
    np.save(f"../temp/{dataset}.weight_occ.npy", weight_occ_pad)
    np.save(f"../temp/{dataset}.graphs_dep.npy", graphs_dep_pad)
    np.save(f"../temp/{dataset}.weight_dep.npy", weight_dep_pad)
    np.save(f"../temp/{dataset}.graphs_mix.npy", graphs_mix_pad)
    np.save(f"../temp/{dataset}.weight_mix.npy", weight_mix_pad)
    np.save(f"../temp/{dataset}.word2vec.npy", word2vec)


if __name__ == '__main__':

    with open(os.sep.join(['..', 'temp', dataset+'.graph.json']), 'r', encoding='utf-8') as f:
        parses = json.load(f)

    construct_graph(parses)
