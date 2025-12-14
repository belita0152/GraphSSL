import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import WeightedRandomSampler

device = torch.device('cuda:0')

class GraphDataset(Dataset):
    """
        edge.py에서 전처리되어 저장된 .pt 파일(x, y, edge)을 로드하여 제공
    """

    def __init__(self, saved_file_path):
        print(f"Loading {saved_file_path}...")
        data = torch.load(saved_file_path, weights_only=False)

        self.x = torch.tensor(data['x']).float()  # Raw EEG
        self.y = torch.tensor(data['y']).long()
        self.edge = torch.tensor(data['edge']).float()  # Adj matrix (PLV)

        self.graph_list = self.process()

    def process(self):
        graph_list = []
        n_samples, n_nodes = self.x.shape[0], self.x.shape[1]

        for i in range(n_samples):
            x_feat = self.x[i]  # (30, 384)

            adj = self.edge[i]  # (30, 30)

            adj.fill_diagonal_(0)  # self loop 제거

            # Edge Index
            edge_index = torch.nonzero(adj, as_tuple=False).t().contiguous() # (N, 2) -> (2, N)

            # Edge Weight
            row, col = edge_index
            edge_attr = adj[row, col]

            graph = Data(x=x_feat, edge_index=edge_index, edge_attr=edge_attr, y=self.y[i])

            graph_list.append(graph)

        return graph_list

    def __getitem__(self, item):
        return self.graph_list[item]

    def __len__(self):
        return len(self.graph_list)


def get_dataloaders(train_path, test_path, batch_size):
    train_dataset, test_dataset = GraphDataset(train_path), GraphDataset(test_path)
    train_dataloader,  test_dataloader = (DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True),
                                           DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True))

    return train_dataloader, test_dataloader



def get_balanced_dataloaders(train_path, test_path, batch_size):
    train_dataset, test_dataset = GraphDataset(train_path), GraphDataset(test_path)

    y_train = train_dataset.y
    if isinstance(y_train, torch.Tensor):
        y_train = y_train.numpy()

    class_counts = np.bincount(y_train)  # attention: (465, 1175), mental: [1026 2257]

    class_weights = 1.0 / class_counts  # 각 class별 가중치 계산 (개수가 적을수록 높은 가중치)

    sample_weights = [class_weights[label] for label in y_train]
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_dataloader, test_dataloader


if __name__ == "__main__":
    import os
    src_path = os.path.join(os.getcwd())
    train_path = os.path.join(src_path, 'mi_train_dataset.pt')
    test_path = os.path.join(src_path, 'mi_test_dataset.pt')

    train_dataloader, test_dataloader = get_dataloaders(train_path, test_path, 32)

    # train_dataloader, test_dataloader = get_balanced_dataloaders(train_path, test_path, 64)

    for batch in train_dataloader:
        print(batch)
