import time
import datetime
import random
import numpy as np
from copy import deepcopy
from sklearn import metrics
from argparse import ArgumentParser
from torch import nn
import torch
from config import args
from dataset import get_data_loader
from model import Model
from utils import setup_logger
from utils import EarlyStopping


def train_eval(cate, loader, model, optimizer, loss_func, device):
    model.train() if cate == "train" else model.eval()
    preds, labels, loss_sum = [], [], 0.
    alphas = []

    for i in range(len(loader)):  # training under mini-batch, backpropagation under large batch
        loss = torch.tensor(0., requires_grad=True).float().to(device)

        for j, graph in enumerate(loader[i]):
            graph = graph.to(device)
            targets = graph.y
            y, a = model(graph)
            loss += loss_func(y, targets)
            preds.append(y.max(dim=1)[1].data)
            labels.append(targets.data)
            alphas.append(a.data)

        loss = loss / len(loader[i])

        if cate == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_sum += loss.data

    preds = torch.cat(preds).tolist()
    labels = torch.cat(labels).tolist()
    loss = loss_sum / len(loader)
    acc = metrics.accuracy_score(labels, preds) * 100
    alphas = torch.cat(alphas).tolist()
    return loss, acc, preds, labels, alphas


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', help='Dataset name.', default='mr')  # mr, ohsumed, R8
    parser.add_argument('--gpu', help='ID of available gpu.', default=0)
    parser.add_argument('--epochs', help='Number of epochs to train.', default=150)  # 200
    parser.add_argument('--batch_size', help='Size of batch for backpropagation.', default=1024)
    parser.add_argument('--mini_batch_size', help='Size of mini-batch for training.', default=16)
    parser.add_argument('--input_dim', help='Dimension of input.', default=300)
    parser.add_argument('--hidden_dim', help='Number of units in hidden layer.', default=96)
    parser.add_argument('--gnn_layer', help='Number of graph layers.', default=2)
    parser.add_argument('--learning_rate', help='Initial learning rate.', default=0.005)
    parser.add_argument('--dropout', help='Dropout rate (1 - keep probability).', default=0.5)
    parser.add_argument('--weight_decay', help='Weight for L2 loss on embedding matrix.', default=0)
    parser.add_argument('--early_stopping', help='Tolerance for early stopping (# of epochs).', default=60)
    parser.add_argument('--not_freeze', help='Not freeze the param of embedding layer.', action='store_true')
    parser.add_argument('--graph_mode', help='Dual graph or single graph.', default='dual')  # dual, single
    parser.add_argument('--rel_type', help='Relation type of single graph.', default='occ')  # occ, dep, mix
    parser.add_argument('--fuse_mode', help='Gate or Attention or static for dual graph fusion.', default='gate')  # gate, atten, static
    parser.add_argument('--alpha', help='Weight of static fusion.', default=0.5)
    parser.add_argument('--share_gru', help='Share the params of GRU.', action='store_true')
    parser.add_argument('--fix_seed', help='Fix the random seed.', action='store_true')
    parser.add_argument('--seed', help='The random seed.', default=123)
    parser.add_argument('--log_dir', help='Log file path', default='log')

    arg = parser.parse_args()

    # params
    dataset = arg.dataset
    gpu_id = arg.gpu
    epoch_num = int(arg.epochs)
    batch_size = int(arg.batch_size)
    mini_batch_size = int(arg.mini_batch_size)
    in_dim = int(arg.input_dim)
    hid_dim = int(arg.hidden_dim)
    gnn_layer = int(arg.gnn_layer)
    lr = float(arg.learning_rate)
    dropout = float(arg.dropout)
    weight_decay = float(arg.weight_decay)
    early_stop_num = int(arg.early_stopping)
    freeze = not arg.not_freeze
    graph_mode = arg.graph_mode
    rel_type = arg.rel_type
    fuse_mode = arg.fuse_mode
    alpha = float(arg.alpha)
    share_gru = arg.share_gru
    fix_seed = arg.fix_seed
    seed = int(arg.seed)
    log_dir = arg.log_dir

    current_time = datetime.datetime.now()
    current_time_str = current_time.strftime("%Y%m%d%H%M%S")
    logger = setup_logger('DGAP', f'{log_dir}/{current_time_str}_{dataset}_{graph_mode}_'
                                  f'{rel_type}_{fuse_mode}_{alpha}_{share_gru}.log')
    logger.info(arg)
    logger.info(f"load dataset: {dataset}.")
    if share_gru:
        logger.info(f"share gru.")
    else:
        logger.info(f"not share gru.")

    # random seed
    if fix_seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        logger.info(f"fix seed: {seed}.")
    else:
        logger.info(f"not fix seed.")

    num_classes = args[dataset]['num_classes']
    (train_loader, valid_loader, test_loader), word2vec = get_data_loader(
        dataset, batch_size, mini_batch_size)
    logger.info(f"train size:{len(train_loader)}, valid size:{len(valid_loader)}, test size:{len(test_loader)}")
    num_words = len(word2vec) - 1  # index of special characters [PAD] in the vocab

    Device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    Model = Model(num_words, num_classes, word2vec=word2vec, in_dim=in_dim, hid_dim=hid_dim,
                  freeze=freeze, step=gnn_layer, dropout=dropout, alpha=alpha,
                  graph_mode=graph_mode, rel=rel_type, fuse_mode=fuse_mode, share_gru=share_gru)
    LossFunc = nn.CrossEntropyLoss()
    Optimizer = torch.optim.Adam(Model.parameters(), lr=lr, weight_decay=weight_decay)

    Model = Model.to(Device)

    logger.info("-" * 50)
    logger.info(f"params: [epoch_num={epoch_num}, batch_size={batch_size}, lr={lr}, "
                f"weight_decay={weight_decay} , dropout={dropout}]")
    logger.info("-" * 50)
    logger.info(Model)
    logger.info("-" * 50)
    logger.info(f"Dataset: {dataset}")

    best_acc = 0.
    best_net = None

    early_stopping = None
    if early_stop_num != -1:
        early_stopping = EarlyStopping(patience=early_stop_num)

    for epoch in range(epoch_num):
        t1 = time.time()
        train_loss, train_acc, _, _, _ = train_eval("train", train_loader, Model, Optimizer, LossFunc, Device)
        valid_loss, valid_acc, _, _, _ = train_eval("valid", valid_loader, Model, Optimizer, LossFunc, Device)
        test_loss, test_acc, _, _, _ = train_eval("test", test_loader, Model, Optimizer, LossFunc, Device)

        if best_acc < valid_acc:  # save the best model on the validation set
            best_acc = valid_acc
            best_net = deepcopy(Model.state_dict())

        cost = time.time() - t1

        logger.info((f"epoch={epoch+1:03d}, cost={cost:.2f}, "
                     f"train:[{train_loss:.4f}, {train_acc:.2f}%], "
                     f"valid:[{valid_loss:.4f}, {valid_acc:.2f}%], "
                     f"test:[{test_loss:.4f}, {test_acc:.2f}%], "
                     f"best_acc={best_acc:.2f}%"))

        # early stopping
        if early_stop_num != -1:
            early_stopping(valid_acc)
            if early_stopping.early_stop:
                logger.info("Early stopping.")
                break

    Model.load_state_dict(best_net)
    test_loss, test_acc, test_preds, test_labels, test_alphas = train_eval("test", test_loader, Model, Optimizer, LossFunc, Device)

    logger.info("Test Precision, Recall and F1-Score...")
    logger.info(metrics.classification_report(test_labels, test_preds, digits=4))
    logger.info("Macro average Test Precision, Recall and F1-Score...")
    logger.info(metrics.precision_recall_fscore_support(test_labels, test_preds, average='macro'))
    logger.info("Micro average Test Precision, Recall and F1-Score...")
    logger.info(metrics.precision_recall_fscore_support(test_labels, test_preds, average='micro'))
