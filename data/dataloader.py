from scipy import io
import numpy as np
import torch
import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader


class Graph_Drowsiness(DGLDataset):
    def __init__(self, name="graph_drowsiness",
                 raw_dir="/data/driver_drowsiness/dataset.mat",
                 threshold=0.5):

        self.base_path = raw_dir
        self.threshold = threshold

        self.ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCZ', 'FC4', 'FT8', 'T3', 'C3', 'Cz',
                         'C4', 'T4', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'A1', 'T5', 'P3', 'PZ', 'P4', 'T6', 'A2', 'O1', 'Oz', 'O2']
        self.sfreq = 128
        self.event_name = ['Alert', 'Drowsy']  # Alert : 0, Drowsy : 1
        self.n_channels = 30
        self.duration = 3
        self.second = 1.5
        self.sfreq = 128  # 전체 길이 = 128 * 3 = 384 time points

        super().__init__(name)

    def process(self):
        matfile = io.loadmat(self.base_path)
        x = matfile['EEGsample']  # EEG data: (2022, 30, 384) = (n_samples, channel, time point)
        y = matfile['substate']  # labels: 0 or 1 (2022,)

        graphs = []

        n_samples, n_nodes = x.shape[0], x.shape[1]

        for i in range(n_samples):
            node_features = x[i]  # (30, 384)

            corr_matrix = np.corrcoef(node_features)  # (30, 384)

            adj = np.abs(corr_matrix) > self.threshold

            np.fill_diagonal(adj, 0)  # self loop 제거

            src, dst = np.where(adj)  # True인 인덱스 추출
            g = dgl.graph((src, dst), num_nodes=n_nodes)

            g.ndata['feat'] = torch.from_numpy(node_features).float()

            weights = np.abs(corr_matrix)[src, dst]
            g.edata['w'] = torch.tensor(weights, dtype=torch.float32)

            graphs.append(g)

        self.graphs = graphs
        self.graph_labels = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.graphs[index], self.graph_labels[index]

    def __len__(self):
        return len(self.graphs)


if __name__ == "__main__":
    dataset = Graph_Drowsiness("graph_drowsiness",
                               raw_dir="/data/driver_drowsiness/dataset.mat",
                               threshold=0.5)

    dataloader = GraphDataLoader(dataset, batch_size=16, shuffle=True)

    for x, y in dataloader:
        print(x, y.shape)




