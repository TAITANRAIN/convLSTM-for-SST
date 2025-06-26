import torch
import xarray as xr
import numpy as np
import os
from model import ConvLSTM
from utils import (
    fill_nan_with_interpolation,
    plot_sst_with_smoothing,
    create_sst_prediction_gif,
    plot_rmse_map,
)
import matplotlib
matplotlib.use('Agg')
def load_data(years, lon_range, lat_range, expand=1.5):
    lon_min, lon_max = lon_range
    lat_min, lat_max = lat_range
    lon_pad = (lon_max - lon_min) * (expand - 1) / 2
    lat_pad = (lat_max - lat_min) * (expand - 1) / 2
    lon_range_ext = (lon_min - lon_pad, lon_max + lon_pad)
    lat_range_ext = (lat_min - lat_pad, lat_max + lat_pad)

    sst_list = []
    for y in years:
        ds = xr.open_dataset(f'sst.day.mean.{y}.nc')
        ds = ds.assign_coords(lon=((ds.lon + 360) % 360))
        sst_y = ds['sst'].sel(
            time=slice(f'{y}-01-01', f'{y}-12-31'),
            lon=slice(*lon_range_ext),
            lat=slice(*lat_range_ext)
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

def plot_sst_with_smoothing(sst_da, date_str):
    import pygmt
    from scipy.ndimage import gaussian_filter

    plot_region = [116, 128, 30, 42]

    sst = sst_da.where(sst_da > -10).squeeze()
    if sst.lat[0] > sst.lat[-1]:
        sst = sst.sortby("lat")

    sst_smooth = gaussian_filter(sst.values, sigma=0.1)

    new_lat = np.linspace(float(sst.lat.min()), float(sst.lat.max()), 1000)
    new_lon = np.linspace(float(sst.lon.min()), float(sst.lon.max()), 1000)
    sst_smooth_xr = xr.DataArray(sst_smooth, coords=sst.coords, dims=sst.dims)
    sst_interp = sst_smooth_xr.interp(lat=new_lat, lon=new_lon, method="linear")
    sst_filled = fill_nan_with_interpolation(sst_interp)

    temp_nc = f"temp_sst_pred_{date_str}.nc"
    sst_filled.to_netcdf(temp_nc)
    vmin, vmax = np.nanpercentile(sst_filled.values, [2, 98])
    pygmt.makecpt(cmap="jet", series=[vmin, vmax, 0.5], continuous=True)

    # vmin, vmax = np.nanpercentile(sst_filled.values, [2, 50])
    # pygmt.makecpt(cmap="jet", series=[12,23, 0.5], continuous=True)

    fig = pygmt.Figure()
    fig.basemap(region=plot_region, projection="M12c", frame=["af", f'+t"SST Prediction {date_str}"'])
    fig.grdimage(grid=temp_nc, cmap=True, nan_transparent=False, interpolation="b")
    fig.coast(region=plot_region, resolution="i", land="gray", shorelines="1/0.5p,black", borders=[1, 2])
    fig.colorbar(frame='af+l"SST (°C)"', position="JBC+w10c/0.4c+h+o0/0.5c")

    out_png = f"sst_prednew_{date_str}.png"
    fig.savefig(out_png, dpi=300)
    fig.show()
    os.remove(temp_nc)




def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_steps = 30  # 过去30天输入
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    lon_range = (116, 128)
    lat_range = (30, 42)

    # 用户交互输入
    start_date_str = input("YY-MM-DD").strip()
    pred_steps = input("predict days").strip()
    try:
        pred_steps = int(pred_steps)
    except:
        print("Error")
        pred_steps = 7


    sst = load_data(years, lon_range, lat_range, expand=1.5)
    sst_array = sst.values.astype(np.float32)
    sst_norm, mean, std = standardize_data(sst_array)

    model = ConvLSTM().to(device)
    model.load_state_dict(torch.load("conv_lstm_sst_model.pth", map_location=device))


    all_times = sst.time.values
    # 找到输入序列的起始索引（要确保有足够的历史数据）
    try:
        start_idx = np.where(all_times == np.datetime64(start_date_str))[0][0]
    except IndexError:
        print("Time error")
        return

    if start_idx < input_steps:

        return


    input_seq_np = sst_norm[start_idx - input_steps : start_idx]
    input_seq = torch.tensor(input_seq_np, dtype=torch.float32).unsqueeze(0).unsqueeze(2)


    preds = predict_multistep(model, input_seq, pred_steps, device)
    preds_real = preds * std + mean

    lat = sst.lat.values
    lon = sst.lon.values
    start_date_np = np.datetime64(start_date_str)
    times = np.array([start_date_np + np.timedelta64(i, 'D') for i in range(pred_steps)])
    preds_xr = xr.DataArray(preds_real, coords=[times, lat, lon], dims=['time', 'lat', 'lon'])
    preds_xr.to_netcdf("sst_prediction_dynamic.nc")


    for i, t in enumerate(preds_xr.time.values):
        date_str = str(t)[:10]
        print(f"{date_str}")
        plot_sst_with_smoothing(preds_xr.isel(time=i), date_str)


if __name__ == "__main__":
    main()