import torch.utils.data as torch_data
import numpy as np
import torch
import pandas as pd


class ForecastDataset(torch_data.Dataset):
    def __init__(self, df, train_window, predict_window, interval=1):
        self.train_window = train_window
        self.interval = interval
        self.predict_window = predict_window
        df = pd.DataFrame(df)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values
        self.data = df
        self.df_length = len(df)
        self.x_end_idx = self.get_x_end_idx()

    def __getitem__(self, index):
        hi = self.x_end_idx[index]
        lo = hi - self.train_window
        train_data = self.data[lo:hi]
        target_data = self.data[hi:hi + self.predict_window]
        x = torch.from_numpy(train_data).type(torch.float)
        y = torch.from_numpy(target_data).type(torch.float)
        return x, y

    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        x_index_set = range(self.train_window, self.df_length - self.predict_window + 1)
        x_end_idx = [x_index_set[j * self.interval] for j in range((len(x_index_set)) // self.interval)]
        return x_end_idx
