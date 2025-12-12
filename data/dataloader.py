import os
import glob
import mne
from scipy import io
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# -----------------------------------------------------------------------------
# We only use dataset with "_2" suffix, recordings of EEG during the mental arithmetic task.
# -----------------------------------------------------------------------------

class AttentionDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.x, self.y = self.get_data(base_path=path)
        self.ch_names = ['Fp1' 'AFF5h' 'AFz' 'F1' 'FC5' 'FC1' 'T7' 'C3' 'Cz' 'CP5' 'CP1' 'P7' 'P3'
                         'Pz' 'POz' 'O1' 'Fp2' 'AFF6h' 'F2' 'FC2' 'FC6' 'C4' 'T8' 'CP2' 'CP6' 'P4'
                         'P8' 'O2' 'HEOG' 'VEOG']
        self.sfreq = 200

    def __getitem__(self, item):
        return torch.tensor(self.x[item]).float(), torch.tensor(self.y[item]).long()

    @staticmethod
    def get_data(base_path):
        total_x, total_y = [], []
        for name in os.listdir(base_path):
            path = os.path.join(base_path, name)
            data = np.load(path)
            x, y = data['x'], data['y']
            total_x.append(x)
            total_y.append(y)
        total_x = np.concatenate(total_x)
        total_y = np.concatenate(total_y)
        return total_x, total_y

    def __len__(self):
        return len(self.y)


class MentalArtihmeticDataset(Dataset):
    def __init__(
            self,
            base_path: str = "/data/mental_arithmetic/physionet.org/files/eegmat/1.0.0"
        ):
        self.data_path = sorted(glob.glob(os.path.join(base_path, "*2.edf")))  # _2 suffix

        info_path = glob.glob(os.path.join(base_path, "*.csv"))[0]
        self.labels = pd.read_csv(info_path)['Count quality']

        self.ch_names = ['EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4', 'EEG F7', 'EEG F8', 'EEG T3', 'EEG T4', 'EEG C3',
                         'EEG C4', 'EEG T5', 'EEG T6', 'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2', 'EEG Fz', 'EEG Cz',
                         'EEG Pz']   # exclude A2-A1 and ECG
        self.event_name = ['G', 'B']  # Good quality : 0, Bad quality : 1
        self.n_channels = 19
        self.second = 10
        self.sfreq = 500

        self.x, self.y = self.parser(self.data_path, self.labels)

    def parser(self, data_path, labels):
        total_x, total_y = [], []
        for i, x_path in enumerate(data_path):
            x = mne.io.read_raw_edf(x_path, preload=False)
            x = x.get_data(picks='eeg')

            x = np.array(x)[:19, :]  # (19, 31000)

            size = self.second * self.sfreq  # 15000
            n_sample = x.shape[1] // size
            x = x[:, :n_sample * size]  # size의 배수 크기로 만들기

            x = x.reshape(-1, len(self.ch_names), size)  # (n_samples, channels, size)

            y = np.zeros((n_sample, ), dtype=int)
            if labels[i] == 1:
                y = np.ones((n_sample,), dtype=int)

            total_x.append(x), total_y.append(y)
        total_x, total_y = np.concatenate(total_x), np.concatenate(total_y)  # (216, 19, 5000)

        return total_x, total_y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        x = torch.tensor(self.x[item], dtype=torch.float)
        y = torch.tensor(self.y[item], dtype=torch.float)

        return x, y


class Drowsiness(Dataset):
    def __init__(
            self,
            base_path: str = "/data/driver_drowsiness/dataset.mat"
        ):

        matfile = io.loadmat(base_path)
        self.x = matfile['EEGsample']  # EEG data: (2022, 30, 384) = (n_samples, channel, time point)
        self.y = matfile['substate']  # labels: 0 or 1 (2022,)

        self.ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCZ', 'FC4', 'FT8', 'T3', 'C3', 'Cz',
                         'C4', 'T4', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'Oz', 'O2']
        self.event_name = ['Alert', 'Drowsy']  # Alert : 0, Drowsy : 1
        self.n_channels = 30
        self.duration = 3
        self.second = 1.5
        self.sfreq = 128  # 전체 길이 = 128 * 3 = 384 time points

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        x = torch.tensor(self.x[item], dtype=torch.float)
        y = torch.tensor(self.y[item], dtype=torch.float)

        return x, y


if __name__ == '__main__':
    driver = Drowsiness()
    dataloader = DataLoader(dataset=driver, batch_size=64, shuffle=True)

    for x, y in dataloader:
        print(x.shape, y.shape)



