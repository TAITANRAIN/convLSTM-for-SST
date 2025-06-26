import xarray as xr
import numpy as np
import pygmt
import os
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

def plot_full_sst_with_land_cover(nc_file, date_str):
    plot_region = [116, 128, 30, 42]
    data_region = [115.5, 135, 29.5, 42.5]

    ds = xr.open_dataset(nc_file)
    target_date = np.datetime64(date_str)
    sub = ds.sel(time=target_date, method='nearest').sel(
        lon=slice(data_region[0], data_region[1]),
        lat=slice(data_region[2], data_region[3])
    )
    sst = sub['sst'].where(sub['sst'] > -10).squeeze()
    if sst.lat[0] > sst.lat[-1]:
        sst = sst.sortby("lat")

    sst_smooth = gaussian_filter(sst.values, sigma=0.2)

    new_lat = np.linspace(float(sst.lat.min()), float(sst.lat.max()), 1000)
    new_lon = np.linspace(float(sst.lon.min()), float(sst.lon.max()), 1000)
    sst_smooth_xr = xr.DataArray(sst_smooth, coords=sst.coords, dims=sst.dims)
    sst_interp = sst_smooth_xr.interp(lat=new_lat, lon=new_lon, method="linear")

    sst_filled = fill_nan_with_interpolation(sst_interp)

    temp_nc = f"temp_sst_full_{date_str}.nc"
    sst_filled.to_netcdf(temp_nc)

    vmin, vmax = np.nanpercentile(sst_filled.values, [2, 98])
    pygmt.makecpt(cmap="jet", series=[vmin, vmax, 0.5], continuous=True)

    fig = pygmt.Figure()
    fig.basemap(region=plot_region, projection="M12c", frame=["af", f'+t"SST {date_str} (OISST V2)"'])
    fig.grdimage(grid=temp_nc, cmap=True, nan_transparent=False, interpolation="b")
    fig.coast(region=plot_region, resolution="i", land="gray", shorelines="1/0.5p,black", borders=[1, 2])
    fig.colorbar(frame='af+l"SST (°C)"', position="JBC+w10c/0.4c+h+o0/0.5c")

    output_file = f"oissfinal_{date_str}.png"
    fig.savefig(output_file, dpi=300)
    fig.show()

    if os.path.exists(temp_nc):
        os.remove(temp_nc)



def plot_multiple_days(nc_file, start_date_str, num_days):
    base_date = np.datetime64(start_date_str)
    for i in range(num_days):
        date_str = str(base_date + np.timedelta64(i, 'D'))[:10]
        print(f"plotting: {date_str}")
        plot_full_sst_with_land_cover(nc_file, date_str)

if __name__ == "__main__":
    nc_file = "./SST.day.mean.2025.nc"
    start_date = "2025-04-16"
    days = 7
    plot_multiple_days(nc_file, start_date, days)
