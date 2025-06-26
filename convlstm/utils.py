import xarray as xr
import numpy as np
import os
import pygmt
import imageio
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata


def fill_nan_with_interpolation(data_array):
    data = data_array.values
    x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    mask = ~np.isnan(data)
    if not mask.any():
        return data_array

    filled = griddata(
        points=(x[mask], y[mask]),
        values=data[mask],
        xi=(x, y),
        method='nearest'
    )
    return xr.DataArray(filled, coords=data_array.coords, dims=data_array.dims)


def plot_sst_with_smoothing(sst_da, date_str):
    plot_region = [116, 128, 30, 42]

    sst = sst_da.where(sst_da > -10).squeeze()
    if sst.lat[0] > sst.lat[-1]:
        sst = sst.sortby("lat")


    sst_smooth = gaussian_filter(sst.values, sigma=0.2)


    new_lat = np.linspace(float(sst.lat.min()), float(sst.lat.max()), 1000)
    new_lon = np.linspace(float(sst.lon.min()), float(sst.lon.max()), 1000)
    sst_smooth_xr = xr.DataArray(sst_smooth, coords=sst.coords, dims=sst.dims)
    sst_interp = sst_smooth_xr.interp(lat=new_lat, lon=new_lon, method="linear")


    sst_filled = fill_nan_with_interpolation(sst_interp)


    temp_nc = f"temp_sst_pred_{date_str}.nc"
    sst_filled.to_netcdf(temp_nc)


    vmin, vmax = np.nanpercentile(sst_filled.values, [2, 98])
    pygmt.makecpt(cmap="jet", series=[vmin, vmax, 0.5], continuous=True)

    fig = pygmt.Figure()
    fig.basemap(region=plot_region, projection="M12c", frame=["af", f'+t"SST Prediction {date_str}"'])
    fig.grdimage(grid=temp_nc, cmap=True, nan_transparent=False, interpolation="b")
    fig.coast(region=plot_region, resolution="i", land="gray",
              shorelines="1/0.5p,black", borders=[1, 2])
    fig.colorbar(frame='af+l"SST (°C)"', position="JBC+w10c/0.4c+h+o0/0.5c")

    out_png = f"sst_pred_{date_str}.png"
    fig.savefig(out_png, dpi=300)
    fig.show()
    os.remove(temp_nc)



def create_sst_prediction_gif(preds_xr, gif_name="sst_prediction_7days.gif", duration=0.8):
    temp_files = []

    for i in range(len(preds_xr.time)):
        date_str = str(preds_xr.time.values[i])[:10]

        plot_sst_with_smoothing(preds_xr.isel(time=i), date_str)
        fname = f"sst_pred_{date_str}.png"
        if os.path.exists(fname):
            temp_files.append(fname)
        else:
            print(f"Error{fname}")

    if not temp_files:
        raise RuntimeError("GIF file not found ")


    with imageio.get_writer(gif_name, mode='I', duration=duration) as writer:
        for file in temp_files:
            image = imageio.imread(file)
            writer.append_data(image)



    for file in temp_files:
        try:
            os.remove(file)
        except Exception as e:
            print(f" {file}: {e}")


def plot_rmse_map(preds_xr, truths_xr, date_list):
    rmse = np.sqrt(((preds_xr - truths_xr) ** 2).mean(dim="time"))
    rmse.name = "rmse"

    temp_nc = "temp_rmse_map.nc"
    rmse.to_netcdf(temp_nc)

    vmin, vmax = 0, float(rmse.max().values)
    pygmt.makecpt(cmap="plasma", series=[vmin, vmax, 0.1], continuous=True)

    fig = pygmt.Figure()
    fig.basemap(region=[116, 128, 30, 42], projection="M12c",
                frame=["af", f'+t"RMSE Map ({date_list[0]}–{date_list[-1]})"'])
    fig.grdimage(grid=temp_nc, cmap=True, nan_transparent=True, interpolation="b")
    fig.coast(region=[116, 128, 30, 42], resolution="i",
              land="gray", shorelines="1/0.5p,black", borders=[1, 2])
    fig.colorbar(frame='af+l"RMSE (°C)"', position="JBC+w10c/0.4c+h+o0/0.5c")

    out_file = f"rmse_map_{date_list[0]}_to_{date_list[-1]}.png"
    fig.savefig(out_file, dpi=300)
    fig.show()
    os.remove(temp_nc)

