import torch
import xarray as xr
import numpy as np
import pygmt
from scipy.ndimage import gaussian_filter
import os
import matplotlib.pyplot as plt
from model import ConvLSTM
from utils import fill_nan_with_interpolation

def load_data(years, lon_range, lat_range):
    sst_list = []
    for y in years:
        ds = xr.open_dataset(f'sst.day.mean.{y}.nc')
        ds = ds.assign_coords(lon=((ds.lon + 360) % 360))
        sst_y = ds['sst'].sel(
            time=slice(f'{y}-01-01', f'{y}-12-31'),
            lon=slice(*lon_range),
            lat=slice(*lat_range)
        ).sortby("lat")
        sst_list.append(sst_y)
    sst = xr.concat(sst_list, dim='time')
    return sst

def standardize_data(sst_array):
    mean = np.nanmean(sst_array)
    std = np.nanstd(sst_array)
    sst_norm = (sst_array - mean) / std
    sst_norm = np.nan_to_num(sst_norm, nan=0.0)
    return sst_norm, mean, std

def predict_multistep(model, input_seq, pred_steps, device):
    model.eval()
    preds = []
    input_seq = input_seq.to(device)
    with torch.no_grad():
        for _ in range(pred_steps):
            pred = model(input_seq)
            preds.append(pred.cpu().numpy().squeeze())
            input_seq = torch.cat([input_seq[:, 1:], pred], dim=1)
    return np.array(preds)

def plot_bias_map(bias_xr, date_str, region):
    bias_smooth = gaussian_filter(bias_xr.values, sigma=0.3)
    new_lat = np.linspace(float(bias_xr.lat.min()), float(bias_xr.lat.max()), 1000)
    new_lon = np.linspace(float(bias_xr.lon.min()), float(bias_xr.lon.max()), 1000)
    bias_interp = bias_xr.interp(lat=new_lat, lon=new_lon, method="linear")
    bias_filled = fill_nan_with_interpolation(bias_interp)

    temp_nc = f"temp_bias_{date_str}.nc"
    bias_filled.to_netcdf(temp_nc)

    vmax = max(abs(bias_filled.values.min()), abs(bias_filled.values.max()))
    pygmt.makecpt(cmap="vik", series=[-vmax, vmax], continuous=True)

    fig = pygmt.Figure()
    fig.basemap(region=region, projection="M12c", frame=["af", f'+t"SST difference {date_str}"'])
    fig.grdimage(grid=temp_nc, cmap=True, nan_transparent=True, interpolation="b")
    fig.coast(region=region, resolution="i", land="gray", shorelines="1/0.5p,black", borders=[1, 2])
    fig.colorbar(frame='af+l"SST difference (°C)"', position="JBC+w10c/0.4c+h+o0/0.5c")

    out_png = f"bias_{date_str}.png"
    fig.savefig(out_png, dpi=300)
    fig.show()

    os.remove(temp_nc)
    print(f"：{out_png}")

def plot_rmse_timeseries(dates, rmses):
    import matplotlib.dates as mdates
    plt.figure(figsize=(10, 5))
    plt.plot(dates, rmses, marker='o', linestyle='-', color='b')
    plt.title("Daily RMSE of SST Prediction")
    plt.xlabel("Date")
    plt.ylabel("RMSE (°C)")
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig("rmse_timeseries.png", dpi=300)
    plt.show()
    print("rmse_timeseries.png")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_steps = 30
    pred_steps = 8

    years = [2025]
    lon_range = (116, 128)
    lat_range = (30, 42)
    region = [116, 128, 30, 42]


    sst_core = load_data(years, lon_range, lat_range)
    sst_array_core = sst_core.values.astype(np.float32)
    sst_norm, mean, std = standardize_data(sst_array_core)
    print(f" mean: {mean:.3f}, std: {std:.3f}")


    sst = load_data(years, lon_range, lat_range)  # 这里不扩展区域
    sst_array = sst.values.astype(np.float32)

    model = ConvLSTM().to(device)
    model.load_state_dict(torch.load("conv_lstm_sst_model.pth", map_location=device))
    print("模型权重已加载")

    all_times = sst.time.values
    input_start_idx = np.where(all_times == np.datetime64("2025-04-15"))[0][0]
    input_end_idx = input_start_idx + input_steps

    input_seq_np = sst_norm[input_start_idx:input_end_idx]
    input_seq = torch.tensor(input_seq_np, dtype=torch.float32).unsqueeze(0).unsqueeze(2)

    predict_dates = np.arange(np.datetime64("2025-04-15"), np.datetime64("2025-04-22") + 1)


    preds = predict_multistep(model, input_seq, pred_steps, device)
    preds_real = preds * std + mean

    lat = sst.lat.values
    lon = sst.lon.values
    preds_xr = xr.DataArray(preds_real, coords=[predict_dates, lat, lon], dims=["time", "lat", "lon"])
    preds_xr.to_netcdf("sst_prediction_15to22.nc")


    truths_xr = sst.sel(time=slice("2025-04-15", "2025-04-22"))

    rmse_list = []
    rmse_dates = []

    for t in predict_dates:
        pred_day = preds_xr.sel(time=t)
        true_day = truths_xr.sel(time=t)
        bias_map = pred_day - true_day
        # 计算当天RMSE
        diff = bias_map.values
        valid_mask = ~np.isnan(diff)
        rmse = np.sqrt(np.mean(diff[valid_mask] ** 2)) if np.any(valid_mask) else np.nan
        rmse_list.append(rmse)
        rmse_dates.append(np.datetime64(t))

        plot_bias_map(bias_map, str(t)[:10], region)

    plot_rmse_timeseries(rmse_dates, rmse_list)

if __name__ == "__main__":
    main()
