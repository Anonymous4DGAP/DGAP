import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric import nn as gnn
import torch.nn.functional as fn


def graph2batch(x, length_list):
    x_list = []
    for graph_len in length_list:
        x_list.append(x[:graph_len])
        x = x[graph_len:]
    x = pad_sequence(x_list, batch_first=True)
    return x


def mask_softmax(x, mask):
    mask_data = x.masked_fill(mask.logical_not(), -1e9)

    return fn.softmax(mask_data, dim=1)


class GRUUint(nn.Module):

    def __init__(self, hid_dim, act, bias=True):
        super(GRUUint, self).__init__()
        self.act = act
        self.lin_z0 = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.lin_z1 = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.lin_r0 = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.lin_r1 = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.lin_h0 = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.lin_h1 = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, a):
        z = (self.lin_z0(a) + self.lin_z1(x)).sigmoid()
        r = (self.lin_r0(a) + self.lin_r1(x)).sigmoid()
        h = self.act((self.lin_h0(a) + self.lin_h1(x * r)))
        return h * z + x * (1 - z)


class GraphLayer(gnn.MessagePassing):

    def __init__(self, hid_dim, dropout=0.5,
                 act=torch.relu, step=2, rel='dep', gru=None):
        super(GraphLayer, self).__init__(aggr='add')
        self.step = step
        self.rel = rel
        self.act = act

        if gru is not None:
            self.gru = gru  # shared gru for dual graph
        else:
            self.gru = GRUUint(hid_dim, act=act)  # separate gru for each graph

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, g):
        for i in range(self.step):
            if self.rel == 'dep':
                a = self.propagate(edge_index=g.edge_index_dep, x=x, edge_attr=self.dropout(g.edge_attr_dep))
            elif self.rel == 'occ':
                a = self.propagate(edge_index=g.edge_index_occ, x=x, edge_attr=self.dropout(g.edge_attr_occ))
            else:  # rel == 'mix'
                a = self.propagate(edge_index=g.edge_index_mix, x=x, edge_attr=self.dropout(g.edge_attr_mix))
            x = self.gru(x, a)

        return x

    def message(self, x_j, edge_attr):
        return x_j * edge_attr.unsqueeze(-1)

    def update(self, inputs):
        return inputs


class FusionLayer(nn.Module):

    def __init__(self, hid_dim, dropout=0.5, alpha=0.5, mode='gate'):
        super(FusionLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        if mode == 'gate':
            self.act = torch.sigmoid
            self.gate = nn.Linear(hid_dim * 2, 1, bias=False)
        elif mode == 'atten':
            self.act = torch.softmax
            self.att = nn.Linear(hid_dim, 1, bias=False)
        self.initial_alpha = alpha  # hyper-parameters for static fusion
        self.register_buffer('alpha', torch.Tensor([alpha]))
        self.mode = mode
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.alpha.data.fill_(self.initial_alpha)

    def forward(self, x_dep, x_occ):

        if self.mode == 'gate':
            # # Gate Mechanism
            x = torch.cat((x_dep, x_occ), dim=-1)
            h = self.act(self.gate(x))
            x = h * x_dep + (1 - h) * x_occ  # (batch_node_num, hidden_dim)

        elif self.mode == 'atten':
            # # Atten Mechanism
            w_dep = fn.elu(self.att(x_dep))
            w_occ = fn.elu(self.att(x_occ))
            h = self.act(torch.cat((w_dep, w_occ), dim=-1), dim=-1)
            x = torch.unsqueeze(h[:, 0], -1) * x_dep + torch.unsqueeze(h[:, 1], -1) * x_occ

        else:  # mode == 'static'
            # Sum Mechanism
            x = self.alpha * x_dep + (1 - self.alpha) * x_occ
            h = torch.ones_like(x) * self.alpha

        # x.shape: (batch_node_num, hidden_dim)
        return x, torch.squeeze(h)


class ReadoutLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.5,
                 act=torch.relu, bias=False):
        super(ReadoutLayer, self).__init__()
        self.act = act
        self.bias = bias
        self.att = nn.Linear(in_dim, 1, bias=False)
        self.emb = nn.Linear(in_dim, in_dim, bias=True)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim, bias=True)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, mask):

        # # Attention Weight Sum - Words
        emb = self.act(self.emb(x))
        att = mask_softmax(self.att(emb), mask)
        x = att * x  # x.shape: (batch_size, graph_node_num, hidden_dim)
        x = torch.sum(x, dim=1)

        # x.shape: (batch_size, hidden_dim)
        x = self.mlp(x)
        return x


class Model(nn.Module):
    def __init__(self, num_words, num_classes, in_dim=300, hid_dim=96, step=2, dropout=0.5, alpha=0.5,
                 word2vec=None, freeze=True, graph_mode='dual', rel='occ', fuse_mode='gate', share_gru=False):
        super(Model, self).__init__()
        if word2vec is None:
            self.embed = nn.Embedding(num_words + 1, in_dim, num_words)
        else:
            self.embed = nn.Embedding.from_pretrained(torch.from_numpy(word2vec).float(), freeze, num_words)
        self.encode = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hid_dim, bias=True)
        )
        self.graph_mode = graph_mode
        self.rel = rel
        if share_gru is True:  # share the parameters of GRU
            gru = GRUUint(hid_dim, act=torch.tanh)
        else:
            gru = None
        if graph_mode == 'single':
            if rel == 'occ':
                self.gcn_occ = GraphLayer(hid_dim, act=torch.tanh, dropout=dropout, step=step, rel='occ')
            elif rel == 'dep':
                self.gcn_dep = GraphLayer(hid_dim, act=torch.tanh, dropout=dropout, step=step, rel='dep')
            else:  # rel == 'mix':
                self.gcn_mix = GraphLayer(hid_dim, act=torch.tanh, dropout=dropout, step=step, rel='mix')
        else:  # mode == 'dual'
            self.gcn_dep = GraphLayer(hid_dim, act=torch.tanh, dropout=dropout, step=step, rel='dep', gru=gru)
            self.gcn_occ = GraphLayer(hid_dim, act=torch.tanh, dropout=dropout, step=step, rel='occ', gru=gru)
            self.fuse = FusionLayer(hid_dim, dropout=dropout, alpha=alpha, mode=fuse_mode)

        self.read = ReadoutLayer(hid_dim, num_classes, act=torch.tanh, dropout=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, g):
        x = self.embed(g.x)
        x = torch.tanh(self.encode(x))
        alpha = torch.ones_like(x)

        if self.graph_mode == 'single':
            if self.rel == 'occ':
                x = self.gcn_occ(x, g)
            elif self.rel == 'dep':
                x = self.gcn_dep(x, g)
            else:
                x = self.gcn_mix(x, g)
        else:  # graph_mode == dual
            x_dep = self.gcn_dep(x, g)  # x_dep.shape: (batch_node_num, hidden_dim)
            x_occ = self.gcn_occ(x, g)  # x_occ.shape: (batch_node_num, hidden_dim)
            x, alpha = self.fuse(x_dep, x_occ)

        # g.length.shape: (batch_size)
        x = graph2batch(x, g.length)  # combine the graph of a mini-batch and restore the batch

        mask = self.get_mask(g)
        x = self.read(x, mask)

        return x, alpha

    def get_mask(self, g):
        mask = pad_sequence([torch.ones(l) for l in g.length], batch_first=True).unsqueeze(-1)
        if g.x.is_cuda:
            mask = mask.cuda(device=g.x.device)
        return mask
