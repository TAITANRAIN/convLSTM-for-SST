
import os
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import ConvLSTM

class SSTDataset(Dataset):
    def __init__(self, sst_array, input_steps=30, pred_steps=7):
        self.sst = sst_array
        self.input_steps = input_steps
        self.pred_steps = pred_steps

    def __len__(self):
        return len(self.sst) - self.input_steps - self.pred_steps + 1

    def __getitem__(self, idx):
        x = self.sst[idx:idx+self.input_steps]
        y = self.sst[idx+self.input_steps:idx+self.input_steps+self.pred_steps]
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        return x, y

def train_val_test_split(data, input_steps, pred_steps):
    total = len(data) - input_steps - pred_steps + 1
    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.1)
    return (0, train_end), (train_end, val_end), (val_end, total)

def get_subset_dataset(sst_norm, input_steps, pred_steps, idx_range):
    start, end = idx_range
    subset = sst_norm[start:end + input_steps + pred_steps - 1]
    return SSTDataset(subset, input_steps, pred_steps)

def main():
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    sst_list = []

    for y in years:
        ds = xr.open_dataset(f'sst.day.mean.{y}.nc')
        ds = ds.assign_coords(lon=((ds.lon + 360) % 360))
        sst_y = ds['sst'].sel(time=slice(f'{y}-01-01', f'{y}-12-31'), lon=slice(116, 128), lat=slice(30, 42)).sortby("lat")
        sst_list.append(sst_y)

    sst = xr.concat(sst_list, dim='time')
    sst_array = sst.values.astype(np.float32)
    mean = np.nanmean(sst_array)
    std = np.nanstd(sst_array)
    sst_norm = (sst_array - mean) / std
    sst_norm = np.nan_to_num(sst_norm, nan=0.0)

    input_steps, pred_steps = 30, 7
    (train_rng, val_rng, test_rng) = train_val_test_split(sst_norm, input_steps, pred_steps)
    train_ds = get_subset_dataset(sst_norm, input_steps, pred_steps, train_rng)
    val_ds = get_subset_dataset(sst_norm, input_steps, pred_steps, val_rng)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    train_rmses, val_rmses = [], []
    epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f'Train Epoch {epoch+1}'):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y[:, 0])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_rmse = (total_loss / len(train_ds)) ** 0.5
        train_rmses.append(train_rmse)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y[:, 0])
                val_loss += loss.item() * x.size(0)
        val_rmse = (val_loss / len(val_ds)) ** 0.5
        val_rmses.append(val_rmse)

        print(f"Epoch {epoch+1}: Train RMSE={train_rmse:.4f}, Val RMSE={val_rmse:.4f}")

    torch.save(model.state_dict(), "conv_lstm_sst_model.pth")


    plt.figure()
    plt.plot(train_rmses, label='Train RMSE')
    plt.plot(val_rmses, label='Val RMSE')
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Training and Validation RMSE")
    plt.grid(True)
    plt.savefig("train_val_rmse_curve.png")


if __name__ == '__main__':
    main()