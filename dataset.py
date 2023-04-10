from config import args
import joblib
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import random
from tqdm import tqdm


class MyDataLoader(object):

    def __init__(self, dataset, batch_size, mini_batch_size=0):
        self.total = len(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        if mini_batch_size == 0:
            self.mini_batch_size = self.batch_size

    def __getitem__(self, item):
        ceil = (item + 1) * self.batch_size
        sub_dataset = self.dataset[ceil - self.batch_size:ceil]
        if ceil >= self.total:
            random.shuffle(self.dataset)
        return DataLoader(sub_dataset, batch_size=self.mini_batch_size)

    def __len__(self):
        if self.total == 0:
            return 0
        return (self.total - 1) // self.batch_size + 1


def split_train_valid_test(data, train_size, valid_part=0.1):
    train_data = data[:train_size]
    test_data = data[train_size:]
    random.shuffle(train_data)
    valid_size = round(valid_part * train_size)
    valid_data = train_data[:valid_size]
    train_data = train_data[valid_size:]
    return train_data, valid_data, test_data


def get_data_loader(dataset, batch_size, mini_batch_size):
    # param
    train_size = args[dataset]["train_size"]

    # load data
    len_input = joblib.load(f"temp/{dataset}.len.inputs.pkl")
    len_graphs_occ = joblib.load(f"temp/{dataset}.len.graphs_occ.pkl")
    len_graphs_dep = joblib.load(f"temp/{dataset}.len.graphs_dep.pkl")
    len_graphs_mix = joblib.load(f"temp/{dataset}.len.graphs_mix.pkl")

    input_word = np.load(f"temp/{dataset}.input_word.npy")
    graphs_occ = np.load(f"temp/{dataset}.graphs_occ.npy")
    weight_occ = np.load(f"temp/{dataset}.weight_occ.npy")
    graphs_dep = np.load(f"temp/{dataset}.graphs_dep.npy")
    weight_dep = np.load(f"temp/{dataset}.weight_dep.npy")
    graphs_mix = np.load(f"temp/{dataset}.graphs_mix.npy")
    weight_mix = np.load(f"temp/{dataset}.weight_mix.npy")

    word2vec = np.load(f"temp/{dataset}.word2vec.npy")
    targets = np.load(f"temp/{dataset}.targets.npy")

    data = []
    for x, y, lx, lo, ld, lm, eo, wo, ed, wd, em, wm in tqdm(list(zip(
            input_word, targets, len_input, len_graphs_occ, len_graphs_dep, len_graphs_mix,
            graphs_occ, weight_occ, graphs_dep, weight_dep, graphs_mix, weight_mix
    )), ascii=True):
        x = torch.tensor(x[:lx], dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        lens = torch.tensor(lx, dtype=torch.long)
        edge_index_occ = torch.tensor(np.array([e[:lo] for e in eo]), dtype=torch.long)
        edge_attr_occ = torch.tensor(wo[:lo], dtype=torch.float)
        edge_index_dep = torch.tensor(np.array([e[:ld] for e in ed]), dtype=torch.long)
        edge_attr_dep = torch.tensor(wd[:ld], dtype=torch.float)
        edge_index_mix = torch.tensor(np.array([e[:lm] for e in em]), dtype=torch.long)
        edge_attr_mix = torch.tensor(wm[:lm], dtype=torch.float)
        data.append(Data(x=x, y=y, length=lens,
                         edge_index_occ=edge_index_occ, edge_attr_occ=edge_attr_occ,
                         edge_index_dep=edge_index_dep, edge_attr_dep=edge_attr_dep,
                         edge_index_mix=edge_index_mix, edge_attr_mix=edge_attr_mix))

    # split
    train_data, valid_data, test_data = split_train_valid_test(data, train_size, valid_part=0.1)

    # return loader & word2vec
    return [MyDataLoader(data, batch_size=batch_size, mini_batch_size=mini_batch_size)
            for data in [train_data, valid_data, test_data]], word2vec

