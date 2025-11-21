import numpy as np
import xarray as xr
import math

# ---------- Paths ----------
obs_file = '/exports/geos.ed.ac.uk/palmer_group/managed/nponomar/edatmo/data-root/L2/by-product/EM27/total_column_qc/QMUM_SN217_251116-251116_L2_V003.nc'
model_file = '/exports/geos.ed.ac.uk/palmer_group/nponomar/icon_runs/work/VPRM_EU_ERA5_22/202501_01_00_0_2160/icon/output/ICON-ART-UNSTRUCTURED_DOM01_20250101T000000Z.nc'
grid_file = '/home/nponomar/icon_Edinburgh/icon_europe_DOM01.nc'

# ---------- Load EM27 observation ----------
ds_obs = xr.open_dataset(obs_file)
obs_time = ds_obs.time.values
obs_lon = ds_obs.proffast_station_londeg.values[~np.isnan(ds_obs.proffast_station_londeg.values)][0]
obs_lat = ds_obs.proffast_station_latdeg.values[~np.isnan(ds_obs.proffast_station_latdeg.values)][0]
XCO2_obs = ds_obs.XCO2_L2.values  # or XCO2_avk if we want AVK-weighted vertical profiles
# ---------- Load ICON-ART model ----------
ds_mod = xr.open_dataset(model_file)
factor = 1e6*28.97/44.  # ppb to ppm conversion if needed
# Mid-layer and top-layer heights
z_ifc = ds_mod.z_ifc.values[1:, :]
z_ifc_top = ds_mod.z_ifc.values[:-1, :]
# Water vapor and hydrometeors
qv = ds_mod.qv[0, :].values   # water vapor
qc = ds_mod.qc[0, :].values   # cloud water
qi = ds_mod.qi[0, :].values   # cloud ice
qr = ds_mod.qr[0, :].values   # rain
qs = ds_mod.qs[0, :].values   # snow
qg = ds_mod.qg[0, :].values   # graupel
# Total moisture fraction for the vertical profile
q_total = qv + qc + qi + qr + qs + qg
# CO2 components
TRCO2 = ds_mod.TRCO2_Anthropogenic_chemtr[0, :].values
BGCO2 = ds_mod.TRCO2_BG_chemtr[0, :].values
RA_CO2 = ds_mod.CO2_RA[0, :].values
GPP_CO2 = ds_mod.CO2_GPP[0, :].values
# Vertical profile for the column with moisture correction
cnc_mod = (TRCO2 + BGCO2 + RA_CO2 - GPP_CO2) * factor / (1 - q_total)

# ---------- Load model grid ----------
ds_grid = xr.open_dataset(grid_file)
lons_rad = ds_grid['clon'].values
lats_rad = ds_grid['clat'].values
len_factor_lat = 110.574
len_factor_lon = 111.320
lons_deg = np.rad2deg(lons_rad)
lats_deg = np.rad2deg(lats_rad)
lons_km = np.array([x*math.cos(lats_rad[ix])*len_factor_lon for ix, x in enumerate(lons_deg)])
lats_km = np.array([x*len_factor_lat for x in lats_deg])
obs_lon_km = obs_lon * math.cos(np.deg2rad(obs_lat)) * len_factor_lon
obs_lat_km = obs_lat * len_factor_lat


# ---------- Horizontal interpolation ----------
distances = np.sqrt((lons_km - obs_lon_km)**2 + (lats_km - obs_lat_km)**2)
num_neigh = 5
if np.any(distances == 0):
    closest_idx = np.where(distances == 0)[0][0]
    cnc_interp_hor = cnc_mod[:, closest_idx]
    z_ifc_hor = z_ifc[:, closest_idx]
else:
    idx_sorted = np.argsort(distances)[:num_neigh]
    weights = 1.0 / distances[idx_sorted]
    weights /= weights.sum()
    cnc_interp_hor = np.nansum(cnc_mod[:, idx_sorted] * weights, axis=1)
    z_ifc_hor = np.nansum(z_ifc[:, idx_sorted] * weights, axis=1)

# ---------- Define model time (instantaneous, e.g., 10:00) ----------
model_time = np.datetime64('2025-11-16T10:00:00') # first model time
time_window = np.timedelta64(30, 'm')  # ±30 min

# ---------- Select and average EM27 observations in the 1-hour window ----------
obs_in_window = ds_obs.sel(time=slice(model_time - time_window, model_time + time_window))
XCO2_obs_avg = obs_in_window.XCO2_L2.mean(dim='time').values
AVK_avg = obs_in_window.XCO2_avk.mean(dim='time').values  # shape: (height_prior,)

# ---------- Vertical interpolation to AVK heights ----------
avk_heights = ds_obs.height_prior.values  # same as AVK height grid
avk_heights_m = avk_heights * 1000  # km → m

# Ensure interpolation array is ascending
def linear_extrapolate(x, xp, fp):
    """Interpolate with linear extrapolation at both ends using two nearest points."""
    # numpy interp for inner points
    y = np.interp(x, xp, fp)
    
    # Bottom linear extrapolation
    mask_bottom = x < xp[0]
    if np.any(mask_bottom):
        slope_bottom = (fp[1] - fp[0]) / (xp[1] - xp[0])
        y[mask_bottom] = fp[0] + slope_bottom * (x[mask_bottom] - xp[0])
    
    # Top linear extrapolation
    mask_top = x > xp[-1]
    if np.any(mask_top):
        slope_top = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        y[mask_top] = fp[-1] + slope_top * (x[mask_top] - xp[-1])
    
    return y
cnc_interp_vert = linear_extrapolate(avk_heights_m, z_ifc_hor[::-1], cnc_interp_hor[::-1])
cnc_interp_vert_no_extrapolation = np.interp(avk_heights_m, z_ifc_hor[::-1], cnc_interp_hor[::-1], 
                                left=np.nan, right=np.nan)
# ---------- Compute XCO2 column using AVK ----------
XCO2_mod = np.nansum(cnc_interp_vert * AVK_avg) / np.sum(AVK_avg)  # weighted sum
XCO2_mod_ne = np.nansum(cnc_interp_vert_no_extrapolation * AVK_avg) / np.sum(AVK_avg)  # weighted sum
# ---------- Compare ----------
print("EM27 observed XCO2 (ppm) [averaged]:", XCO2_obs_avg)
print("Model XCO2 column (ppm):", XCO2_mod)
print("Model XCO2 column (ppm) no extrapolation:", XCO2_mod_ne)