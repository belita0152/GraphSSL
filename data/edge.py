import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import mne
from mne_connectivity import spectral_connectivity_epochs


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCZ', 'FC4', 'FT8', 'T3', 'C3', 'Cz',
                         'C4', 'T4', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'Oz', 'O2']
        self.sfreq = 128

        self.x = x
        self.y = y
        self.edge = self.get_edge(self.x)

    def get_edge(self, x):
        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types='eeg')
        fmin, fmax = 8., 13.

        edges_list = []
        n_samples = x.shape[0]

        for i in range(n_samples):
            tmp_epoch = mne.EpochsArray(x[i:i+1], info=info)

            con = spectral_connectivity_epochs(
                tmp_epoch,
                method='plv',
                mode='multitaper',
                sfreq=self.sfreq,
                fmin=fmin, fmax=fmax,
                faverage=True,  # 주파수 대역 평균
                verbose=False
            )

            con_data = con.get_data(output='dense')  # (n_channels, n_channels)
            con_data = np.nan_to_num(con_data[:, :, 0])  # 마지막 차원은 주파수 평균이라 0번 인덱스
            np.fill_diagonal(con_data, 0)

            con_data[con_data < 0.3] = 0  # threshold
            edges_list.append(con_data)

        edges = np.array(edges_list)

        return edges

    def __getitem__(self, item):
        x = torch.tensor(self.x[item], dtype=torch.float)
        y = torch.tensor(self.y[item], dtype=torch.long)
        edge = torch.tensor(self.edge[item], dtype=torch.long)
        x, y, edge = x.to(device), y.to(device), edge.to(device)
        return x, y, edge

    def __len__(self):
        return len(self.x)


if __name__ == "__main__":
    from dataloader import AttentionDataset, MentalArtihmeticDataset, DrowsinessDataset
    from utils import load_data

    device = torch.device('cuda:0')
    batch_size = 64
    save_path1, save_path2 = "train_proc_edge.pt", "test_proc_edge.pt"

    driver = DrowsinessDataset()
    x, y = driver.x, driver.y

    (train_x_, train_y_), (test_x_, test_y_) = load_data(x, y, split_ratio=0.8)

    train_dataset = CustomDataset(train_x_, train_y_)
    train_x, train_y, train_edge = train_dataset.x, train_dataset.y, train_dataset.edge

    test_dataset = CustomDataset(test_x_, test_y_)
    test_x, test_y, test_edge = test_dataset.x, test_dataset.y, test_dataset.edge

    train_edge = torch.tensor(train_edge, dtype=torch.float)
    test_edge = torch.tensor(test_edge, dtype=torch.float)

    torch.save({'x': train_x, 'y': train_y, 'edge': train_edge}, save_path1)
    torch.save({'x': test_x_, 'y': test_y, 'edge': test_edge}, save_path2)
    print(f"저장 완료: {save_path1}")
    print(f"저장 완료: {save_path2}")





