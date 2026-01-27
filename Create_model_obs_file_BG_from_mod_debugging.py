import os
import glob
import numpy as np
import xarray as xr
import math
from multiprocessing import Pool
import re 
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
# ---------- Paths ----------
obs_dir = '/exports/geos.ed.ac.uk/palmer_group/managed/nponomar/edatmo/data-root/L2/by-product/EM27/total_column_qc/'
model_dir = '/exports/geos.ed.ac.uk/palmer_group/nponomar/icon_runs/work/VPRM_EU_ERA5_22/202504_01_00_0_2183/icon/output/'
grid_file = '/home/nponomar/icon_Edinburgh/icon_europe_DOM01.nc'
default_obs_version = '*V014*'
outpath = '/exports/geos.ed.ac.uk/palmer_group/nponomar/EM27/EM27_ICONART_preprocessing/'
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

# ---------- Helper Functions ----------
def linear_extrapolate(x, xp, fp):
    y = np.interp(x, xp, fp)
    mask_bottom = x < xp[0]
    if np.any(mask_bottom):
        slope_bottom = (fp[1] - fp[0]) / (xp[1] - xp[0])
        y[mask_bottom] = fp[0] + slope_bottom * (x[mask_bottom] - xp[0])
    mask_top = x > xp[-1]
    if np.any(mask_top):
        slope_top = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        y[mask_top] = fp[-1] + slope_top * (x[mask_top] - xp[-1])
    return y

def get_file_pairs(model_dir, obs_dir, obs_v=default_obs_version):
    model_files = sorted(glob.glob(os.path.join(model_dir, '*.nc')))
    obs_files = sorted(glob.glob(os.path.join(obs_dir, obs_v + '.nc')))

    print("Model examples:", model_files[:3])
    print("Obs   examples:", obs_files[:3])

    file_pairs = []

    # Regex for obs files: extract the 6-digit date before the dash
    obs_dates = {}
    for of in obs_files:
        m = re.search(r'_(\d{6})-', os.path.basename(of))
        if m:
            obs_dates[of] = m.group(1)   # format: DDMMYY

    for mf in model_files:
        base = os.path.basename(mf)
        # Extract YYYYMMDD from model filename
        m = re.search(r'_(\d{8})T', base)
        if not m:
            continue

        yyyymmdd = m.group(1)
        yyyy = yyyymmdd[:4]
        mm   = yyyymmdd[4:6]
        dd   = yyyymmdd[6:8]

        # Convert to DDMMYY
        ddmmyy = yyyy[2:] + mm + dd

        # Match with obs dates
        for of, d_obs in obs_dates.items():
            if d_obs == ddmmyy:
                file_pairs.append((mf, of))

    return file_pairs

def get_pid_from_filename(obs_file):
    """
    Extracts the station code (PID) from an observation filename.
    Example: 'JCMB_SN219_250325-250325_L2_V009.nc' -> 'JCMB_SN219'
    """
    basename = os.path.basename(obs_file)
    m = re.match(r'([A-Z]+_SN\d+)_', basename)

    return m.group(1)

def get_unique_obs_files_from_pairs(file_pairs):
    unique_files = OrderedDict()
    
    for _, obs_file in file_pairs:
        pid = get_pid_from_filename(obs_file)
        if pid is None:
            continue
        if pid not in unique_files:
            unique_files[pid] = obs_file

    return list(unique_files.values())



class Station:
    """
    Precompute horizontal & vertical interpolation info for a station.
    """
    def __init__(self, obs_file, lons_km, lats_km, num_neigh=5):
        ds_obs = xr.open_dataset(obs_file)

        mask = ~np.isnan(ds_obs.londeg.values)
        self.pid = get_pid_from_filename(obs_file)
        lon = ds_obs.londeg.values[mask][0]
        lat = ds_obs.latdeg.values[mask][0]
        self.height_agl = ds_obs.station_height_above_ground.values[mask][0]

        self.lon_km = lon * math.cos(np.deg2rad(lat)) * 111.320
        self.lat_km = lat * 110.574

        # ---------- Horizontal interpolation ----------
        distances = np.sqrt((lons_km - self.lon_km)**2 + (lats_km - self.lat_km)**2)
        if np.any(distances == 0):
            self.hor_idx = np.array([np.where(distances==0)[0][0]])
            self.hor_weights = np.array([1.0])
        else:
            idx_sorted = np.argsort(distances)[:num_neigh]
            weights = 1.0 / distances[idx_sorted]
            weights /= weights.sum()
            self.hor_idx = idx_sorted
            self.hor_weights = weights

        # ---------- Vertical interpolation ----------
        self.avk_heights_m = ds_obs.height_prior.values * 1000  # km → m
        self.topo_obs = ds_obs.altim.values - self.height_agl
        if np.any(~np.isnan(self.topo_obs)):
            self.topo_obs = self.topo_obs[~np.isnan(self.topo_obs)][0]
            self.valid = True
        else:
            print('!!!!!!!!!!', self.pid, ds_obs.altim,  ds_obs.station_height_above_ground)
            self.topo_obs = 0
            self.valid = False
        ds_obs.close()
# ---------- Horizontal interpolation ----------
def hor_interp(arr, station):
    if station.hor_weights.size > 1:
        return np.nansum(arr[:, station.hor_idx] * station.hor_weights, axis=1)
    else:
        '!!!ERROR NO INTERP. WEIGHTS!!!'




file_pairs = get_file_pairs(model_dir, obs_dir, obs_v='*V014*')

# Precompute station info from the first obs file

unique_obs_files = get_unique_obs_files_from_pairs(file_pairs)
# stations = {get_pid_from_filename(f): Station(f, lons_km, lats_km) for f in unique_obs_files}
stations = {}
for f in unique_obs_files:
    s = Station(f, lons_km, lats_km)
    if s.valid:
        stations[s.pid] = s
    else:
        print(f"IGNORED STATION: {f}")


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


def debug_plot_stepwise(
        z_model, cnc_model,
        z_interp, cnc_interp,
        avk, pid, timestamp,
        cnc_model_comp,     
        cnc_interp_comp,     
        comp_name,

        outdir="debug_plots"
    ):

    os.makedirs(outdir, exist_ok=True)
    XCO2_mod_total = np.nansum(cnc_interp * avk) / np.sum(avk)
    XCO2_mod_comp  = np.nansum(cnc_interp_comp * avk) / np.sum(avk)
    fig, axs = plt.subplots(2, 4, figsize=(22, 10))

    axs[0,0].plot(cnc_model, z_model, 'k.-')
    axs[0,0].set_title("Raw model profile (TOTAL)")
    axs[0,0].set_xlabel("CO₂ (ppm)")
    axs[0,0].set_ylabel("z (m)")

    axs[0,1].plot(cnc_interp, z_interp, 'b.-')
    axs[0,1].set_title("Interpolated (TOTAL)")
    axs[0,1].set_xlabel("CO₂ (ppm)")
    axs[0,1].set_ylabel("z (m)")

    axs[0,2].plot(avk, z_interp, 'r.-')
    axs[0,2].set_title("Averaging Kernel")
    axs[0,2].set_xlabel("AK")
    axs[0,2].set_ylabel("z (m)")

    contrib_total = cnc_interp * avk
    axs[0,3].scatter(contrib_total, z_interp, s=40)
    axs[0,3].set_title("Contribution (TOTAL)")
    axs[0,3].set_xlabel("cnc * AK")
    axs[0,3].legend([
        f"XCO₂_total_model = Σ(cnc·AK) / Σ(AK) = {XCO2_mod_total:.2f} ppm"
    ], loc="upper right")

    axs[1,0].plot(cnc_model_comp, z_model, '.-')
    axs[1,0].set_title(f"Raw model profile ({comp_name})")
    axs[1,0].set_xlabel("CO₂ (ppm)")
    axs[1,0].set_ylabel("z (m)")

    axs[1,1].plot(cnc_interp_comp, z_interp, '.-')
    axs[1,1].set_title(f"Interpolated ({comp_name})")
    axs[1,1].set_xlabel("CO₂ (ppm)")
    axs[1,1].set_ylabel("z (m)")

    axs[1,2].plot(avk, z_interp, 'r.-')
    axs[1,2].set_title("Averaging Kernel")
    axs[1,2].set_xlabel("AK")
    axs[1,2].set_ylabel("z (m)")

    contrib_comp = cnc_interp_comp * avk
    axs[1,3].scatter(contrib_comp, z_interp, s=40)
    axs[1,3].set_title(f"Contribution ({comp_name})")
    axs[1,3].set_xlabel("cnc * AK")
    axs[1,3].legend([
        f"XCO₂_({comp_name})_model = Σ(cnc·AK) / Σ(AK) = {XCO2_mod_comp:.2f} ppm"
    ], loc="upper right")

    fig.suptitle(f"{pid} – {timestamp} – Component {comp_name}")
    plt.tight_layout()

    fname = os.path.join(outdir, f"debug_{pid}_{timestamp}_{comp_name}.png")
    plt.savefig(fname)
    plt.close()
    print(fname)


def process_file_pair_full(file_pair, stations, min_nobs=7):
    """
    Process a single model-obs file pair and return a dict with:
    - pid
    - model_time
    - XCO2_obs
    - XCO2_mod
    - XCO2_mod_no_extrap
    - interpolated model components: TRCO2, BGCO2, RA, GPP
    """
    
    model_file, obs_file = file_pair

    pid = get_pid_from_filename(obs_file)
    if pid not in stations:
        print(f"Skipping {pid}: no valid Station info")
        return None
    station = stations[pid]
    print('Processing', model_file, obs_file, pid)
    # Open datasets
    ds_mod = xr.open_dataset(model_file)
    ds_obs = xr.open_dataset(obs_file)

    # ---------- Model CO2 ----------
    factor = 1e6*28.97/44.
    z_ifc = ds_mod.z_mc.values[:] - ds_mod.topography_c.values #ds_mod.z_ifc.values[1:, :]
    q_total = ds_mod.qv[0].values + ds_mod.qc[0].values + ds_mod.qi[0].values + \
              ds_mod.qr[0].values + ds_mod.qs[0].values + ds_mod.qg[0].values
    model_layers = ds_mod.z_ifc.values[1:, :]
    TRCO2 = ds_mod.TRCO2_Anthropogenic_chemtr[0].values * factor / (1 - q_total)
    TRCO2_BG = ds_mod.TRCO2_BG_chemtr[0].values * factor / (1 - q_total)
    RA_CO2 = ds_mod.CO2_RA[0].values * factor / (1 - q_total)
    GPP_CO2 = ds_mod.CO2_GPP[0].values * factor / (1 - q_total)
    cnc_total = TRCO2 + TRCO2_BG + RA_CO2 - GPP_CO2

    cnc_hor_total = hor_interp(cnc_total, station)
    cnc_hor_TR = hor_interp(TRCO2, station)
    cnc_hor_TR_BG = hor_interp(TRCO2_BG, station)
    cnc_hor_RA = hor_interp(RA_CO2, station)
    cnc_hor_GPP = hor_interp(GPP_CO2, station)
    z_ifc_hor = hor_interp(z_ifc, station)
    mod_topo_hor = hor_interp(model_layers, station)

    print('Model topo', mod_topo_hor[-1], 'Observation topo ', station.topo_obs)
    # ---------- Vertical interpolation ----------
    avk_heights_total = station.avk_heights_m  - station.topo_obs#+ station.height_agl
    # ---------- no extrapolation ----------
    # cnc_vert = np.interp(avk_heights_total, z_ifc_hor[::-1], cnc_hor_total[::-1],
    #                      left=np.nan, right=np.nan)
    # cnc_vert_TR = np.interp(avk_heights_total, z_ifc_hor[::-1], cnc_hor_TR[::-1],
    #                         left=np.nan, right=np.nan)
    # cnc_vert_BG = np.interp(avk_heights_total, z_ifc_hor[::-1], cnc_hor_BG[::-1],
    #                         left=np.nan, right=np.nan)
    # cnc_vert_RA = np.interp(avk_heights_total, z_ifc_hor[::-1], cnc_hor_RA[::-1],
    #                         left=np.nan, right=np.nan)
    # cnc_vert_GPP = np.interp(avk_heights_total, z_ifc_hor[::-1], cnc_hor_GPP[::-1],
    #                          left=np.nan, right=np.nan)
    cnc_vert = linear_extrapolate(avk_heights_total, z_ifc_hor[::-1], cnc_hor_total[::-1])
    cnc_vert_TR = linear_extrapolate(avk_heights_total, z_ifc_hor[::-1], cnc_hor_TR[::-1])
    cnc_vert_BG = linear_extrapolate(avk_heights_total, z_ifc_hor[::-1], cnc_hor_TR_BG[::-1])
    cnc_vert_RA = linear_extrapolate(avk_heights_total, z_ifc_hor[::-1], cnc_hor_RA[::-1])
    cnc_vert_GPP = linear_extrapolate(avk_heights_total, z_ifc_hor[::-1], cnc_hor_GPP[::-1])

    # ---------- Model time ----------
    basename = os.path.basename(model_file)
    m = re.search(r'_(\d{8}T\d{6})Z', basename)  # extract YYYYMMDDTHHMMSS
    date_str = m.group(1)  # '20250325T170000'

    # Parse into numpy.datetime64
    model_time = np.datetime64(
        f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T{date_str[9:11]}:{date_str[11:13]}:{date_str[13:15]}"
    )
    time_window = np.timedelta64(30, 'm')

    # ---------- Observations ----------
    obs_in_window = ds_obs.sel(time=slice(model_time - time_window, model_time + time_window))
    if len(obs_in_window.time)<min_nobs:
        XCO2_obs_avg = np.nan
    else:
        XCO2_obs_avg = obs_in_window.XCO2.mean(dim='time').values
    AVK_avg = obs_in_window.XCO2_AK.mean(dim='time').values
    
    avk_present = False
    if np.any(~np.isnan(AVK_avg)):
        avk_present = True    
    # Vertical AVK-weighted
    XCO2_mod = np.nansum(cnc_vert * AVK_avg) / np.sum(AVK_avg)
    XCO2_mod_TR = np.nansum(cnc_vert_TR * AVK_avg) / np.sum(AVK_avg)
    XCO2_mod_BG = np.nansum(cnc_vert_BG * AVK_avg) / np.sum(AVK_avg)
    XCO2_mod_RA = np.nansum(cnc_vert_RA * AVK_avg) / np.sum(AVK_avg)
    XCO2_mod_GPP = np.nansum(cnc_vert_GPP * AVK_avg) / np.sum(AVK_avg)

    print("Model levels (z_ifc_hor):", z_ifc_hor[::-1])
    print("AVK heights total:", avk_heights_total)
    print("Model TR concentration:", cnc_hor_TR[::-1])
    print("Model interpolated TR concentration:", cnc_vert_TR)
    print('Avk heights', station.avk_heights_m)
    print('Avk agl', station.height_agl)

    # if np.any(~np.isnan(XCO2_mod)):
    #     debug_plot_stepwise(
    #         z_model = z_ifc_hor[::-1],
    #         cnc_model = cnc_hor_total[::-1],
    #         z_interp = avk_heights_total,
    #         cnc_interp = cnc_vert,
    #         avk = AVK_avg,
    #         pid = pid,
    #         timestamp = str(model_time),

    #         cnc_model_comp = cnc_hor_TR[::-1],
    #         cnc_interp_comp = cnc_vert_TR,
    #         comp_name = "Anthr"
    #     )
    #     debug_plot_stepwise(
    #         z_model = z_ifc_hor[::-1],
    #         cnc_model = cnc_hor_total[::-1],
    #         z_interp = avk_heights_total,
    #         cnc_interp = cnc_vert,
    #         avk = AVK_avg,
    #         pid = pid,
    #         timestamp = str(model_time),

    #         cnc_model_comp = cnc_hor_total[::-1],
    #         cnc_interp_comp = cnc_vert,
    #         comp_name = "Total"
    #     )
    #     debug_plot_stepwise(
    #         z_model = z_ifc_hor[::-1],
    #         cnc_model = cnc_hor_total[::-1],
    #         z_interp = avk_heights_total,
    #         cnc_interp = cnc_vert,
    #         avk = AVK_avg,
    #         pid = pid,
    #         timestamp = str(model_time),

    #         cnc_model_comp = cnc_hor_TR_BG [::-1],
    #         cnc_interp_comp = cnc_vert_BG,
    #         comp_name = "BG"
    #     )


    # Close datasets
    ds_mod.close()
    ds_obs.close()
    # ---------- Add profile dictionary ----------
    profile_dict = {
    'cnc_vert_total': cnc_vert,
    'cnc_vert_Anthr': cnc_vert_TR,
    'cnc_vert_BG': cnc_vert_BG,
    'cnc_vert_RA': cnc_vert_RA,
    'cnc_vert_GPP': cnc_vert_GPP,
    'avk_heights': avk_heights_total,
    'AVK_avg':AVK_avg
    }

    return {
        'pid': pid,
        'time': model_time,
        'XCO2_obs': XCO2_obs_avg,
        'XCO2_mod': XCO2_mod,
        'XCO2_mod_TR': XCO2_mod_TR,
        'XCO2_mod_BG': XCO2_mod_BG,
        'XCO2_mod_RA': XCO2_mod_RA,
        'XCO2_mod_GPP': XCO2_mod_GPP,
        'model_topo': mod_topo_hor[-1],
        'n_obs': len(obs_in_window.time),
        'avk_present': avk_present,
        'obs_topo': station.topo_obs,
        'profile_dict': profile_dict
    }
# ---------- Main ----------

args = [(fp, stations) for fp in file_pairs]

with Pool(4) as pool:
    all_results = pool.starmap(process_file_pair_full, args)
# for i in range(40):
#     print(i, args[i][0], args[i][1])
#     process_file_pair_full(args[i][0], args[i][1])
# Build sorted unique times and pids
# times = sorted(list({d['time'] for d in all_results}))
# pids  = sorted(list({d['pid']  for d in all_results}))
# Filter out None results
all_results_valid = [d for d in all_results if d is not None and 'time' in d and 'pid' in d]

# Build sorted unique times and pids from valid results only
times = sorted({d['time'] for d in all_results_valid})
pids  = sorted({d['pid']  for d in all_results_valid})

NT = len(times)
NS = len(pids)

# Prepare index maps
time_idx = {t: i for i, t in enumerate(times)}
pid_idx  = {p: i for i, p in enumerate(pids)}




# Initialize arrays (much cleaner: only store arrays!)
data_dict = {
    'XCO2_obs':     np.full((NT, NS), np.nan),
    'XCO2_mod':     np.full((NT, NS), np.nan),
    'XCO2_mod_A':  np.full((NT, NS), np.nan),
    'XCO2_mod_BG':  np.full((NT, NS), np.nan),
    'XCO2_mod_RA':  np.full((NT, NS), np.nan),
    'XCO2_mod_GPP': np.full((NT, NS), np.nan),
    'model_topo':   np.full((NT, NS), np.nan),
    'obs_topo':     np.full((NT, NS), np.nan)
}

# Fill arrays
# for d in all_results:
for d in all_results_valid:
    ti = time_idx[d['time']]
    si = pid_idx[d['pid']]
    data_dict['XCO2_obs'    ][ti, si] = d['XCO2_obs']
    data_dict['XCO2_mod'    ][ti, si] = d['XCO2_mod']
    data_dict['XCO2_mod_A' ][ti, si] = d['XCO2_mod_TR']
    data_dict['XCO2_mod_BG' ][ti, si] = d['XCO2_mod_BG']
    data_dict['XCO2_mod_RA' ][ti, si] = d['XCO2_mod_RA']
    data_dict['XCO2_mod_GPP'][ti, si] = d['XCO2_mod_GPP']
    data_dict['model_topo'][ti, si] = d['model_topo']
    data_dict['obs_topo'][ti, si]   = d['obs_topo']
# Create xarray dataset
ds = xr.Dataset(
    {k: (('time', 'pid'), v) for k, v in data_dict.items()},
    coords={'time': times, 'pid': pids}
)


# --- Build output filename from first/last time ---
t0 = np.datetime_as_string(ds.time.min().values, unit='m')
t1 = np.datetime_as_string(ds.time.max().values, unit='m')

# Convert to compact format: 2025-01-25T00:00 → 20250125T0000
t0 = t0.replace('-', '').replace(':', '')
t1 = t1.replace('-', '').replace(':', '')

outfile = outpath + f"processed_EM27_ICONART_{t0}_{t1}_modBG.nc"

print("Writing:", outfile)
ds.to_netcdf(outfile)






def plot_hourly_month(ds, pid, year, month, outdir=outpath + "Mod_BG"):
    os.makedirs(outdir, exist_ok=True)

    # Select PID and month
    sel = ds.sel(pid=pid)

    # Get model and observation topography (mean over the month)
    model_topo = sel["model_topo"].values[~np.isnan(sel["model_topo"].values)][0]
    obs_topo = sel["obs_topo"].values[~np.isnan(sel["obs_topo"].values)][0]

    time_sel = sel.time.dt.year == year
    time_sel &= sel.time.dt.month == month
    d = sel.sel(time=time_sel)

    print(f'Plotting data for {pid} {year}-{month:02d}')
    if len(d.time) == 0:
        print(f"No data for {pid} {year}-{month:02d}")
        return

    obs  = d["XCO2_obs"].values
    mod  = d["XCO2_mod"].values
    bg   = d["XCO2_mod_BG"].values
    anth = d["XCO2_mod_A"].values  # Anthr
    ra   = d["XCO2_mod_RA"].values
    gpp  = d["XCO2_mod_GPP"].values
    times = d.time.values
    if np.all(np.isnan(obs)) and np.all(np.isnan(mod)):
        print(f"All data NaN for {pid} {year}-{month:02d}, skipping plot")
        return
    # Compute mean/median for legend
    def stats(x):
        return f"{np.nanmean(x):.2f}/{np.nanmedian(x):.2f}"

    stats_dict = {
        "OBS": stats(obs),
        "OBS_enchancement": stats(obs-bg),
        "MOD": stats(mod),
        "BG": stats(bg),
        "Anth": stats(anth),
        "RA": stats(ra),
        "GPP": stats(gpp)
        
    }

    plt.figure(figsize=(12,5))
    plt.plot(times, obs, 'k-', lw=1.5, label=f"OBS ({stats_dict['OBS']} / stats)")
    plt.plot(times, mod, 'r-', lw=1.5, label=f"MOD ({stats_dict['MOD']})")

    # Stacked shaded areas
    plt.fill_between(times, bg, 410, color='cyan', alpha=0.5, label=f"BG ({stats_dict['BG']})")
    plt.fill_between(times, bg, bg+anth, color='gray', alpha=0.5, label=f"Anthr ({stats_dict['Anth']})")
    plt.fill_between(times, bg+anth, bg+anth+ra, color='olive', alpha=0.5, label=f"RA ({stats_dict['RA']})")
    plt.fill_between(times, bg+anth+ra, bg+anth+ra+gpp, color='lime', alpha=0.5, label=f"GPP ({stats_dict['GPP']})")

    plt.title(f"Hourly XCO2 - {pid} {year}-{month:02d} | Model topo: {model_topo:.1f} m, Obs topo: {obs_topo:.1f} m | Obs. enchancement {stats_dict['OBS_enchancement']}")
    plt.ylabel("XCO2 [ppm]")
    plt.ylim(bottom=410)
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True)

    # Save
    fname = os.path.join(outdir, f"{pid}_{year}{month:02d}.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

# --- Call for all PIDs and months ---
all_times = pd.to_datetime(ds.time.values)
months = sorted(set((t.year, t.month) for t in all_times))

for pid in ds.pid.values:
    for year, month in months:
        plot_hourly_month(ds, pid, year, month)


def compute_monthly_metrics(ds):
    """Compute bias, CRMSE, correlation, mean per site and month using numpy only."""
    metrics_site = {}
    metrics_site_full = {}
    metrics_month = {}
    metrics_month_full = {}

    all_times = pd.to_datetime(ds.time.values)
    months = sorted(set((t.year, t.month) for t in all_times))

    for pid in ds.pid.values:
        metrics_site[pid] = {"bias": [], "rmse": [], "corr": [], "mean": []}
        metrics_site_full[pid] = {"mean_obs": [], "mean_mod": []}

    for year, month in months:
        for pid in ds.pid.values:
            sel = ds.sel(pid=pid)
            time_sel = (sel.time.dt.year == year) & (sel.time.dt.month == month)
            d = sel.sel(time=time_sel)
            obs = d["XCO2_obs"].values #- d['XCO2_mod_BG'].values
            mod = d["XCO2_mod"].values

            mask = ~np.isnan(obs) & ~np.isnan(mod)
            if np.sum(mask) == 0:
                continue

            obs_valid = obs[mask]
            mod_valid = mod[mask]

            bias = np.mean(mod_valid - obs_valid)

            # Centered RMSE (CRMSE)
            obs_c = obs_valid - np.mean(obs_valid)
            mod_c = mod_valid - np.mean(mod_valid)
            crmse = np.sqrt(np.mean((mod_c - obs_c)**2))

            # Pearson correlation using numpy
            corr = np.corrcoef(obs_valid, mod_valid)[0, 1] if len(obs_valid) > 1 else np.nan

            mean_obs = np.mean(obs_valid)
            mean_mod = np.mean(mod_valid)

            # Save per site
            metrics_site[pid]["bias"].append(bias)
            metrics_site[pid]["rmse"].append(crmse)
            metrics_site[pid]["corr"].append(corr)
            metrics_site[pid]["mean"].append(mean_obs)
            metrics_site_full[pid]["mean_obs"].append(mean_obs)
            metrics_site_full[pid]["mean_mod"].append(mean_mod)

            # Save per month
            metrics_month.setdefault((year, month), {"bias": [], "rmse": [], "corr": [], "mean": []})
            metrics_month_full.setdefault((year, month), {"mean_obs": [], "mean_mod": []})

            metrics_month[(year, month)]["bias"].append(bias)
            metrics_month[(year, month)]["rmse"].append(crmse)
            metrics_month[(year, month)]["corr"].append(corr)
            metrics_month[(year, month)]["mean"].append(mean_obs)

            metrics_month_full[(year, month)]["mean_obs"].append(mean_obs)
            metrics_month_full[(year, month)]["mean_mod"].append(mean_mod)

    # Average metrics over months/sites
    metrics_site_avg = {pid: {k: np.nanmean(v) for k, v in metrics_site[pid].items()} for pid in metrics_site}
    metrics_month_avg = {m: {k: np.nanmean(v) for k, v in metrics_month[m].items()} for m in metrics_month}

    return metrics_site_avg, metrics_month_avg, metrics_site_full, metrics_month_full


def annotate_bars(ax, values):
    for i, v in enumerate(values):
        ax.text(i, v + 0.01 * max(values), f"{v:.2f}", ha='center', va='bottom', fontsize=9)


def plot_metrics_site(metrics_site_avg, metrics_site_full, outdir):
    """Plot metrics per site with Bias, CRMSE, Correlation, and Mean OBS/MOD."""
    os.makedirs(outdir, exist_ok=True)
    pids = list(metrics_site_avg.keys())

    bias  = [metrics_site_avg[p]["bias"] for p in pids]
    rmse  = [metrics_site_avg[p]["rmse"] for p in pids]
    corr  = [metrics_site_avg[p]["corr"] for p in pids]
    mean_obs = [np.nanmean(metrics_site_full[p]["mean_obs"]) for p in pids]
    mean_mod = [np.nanmean(metrics_site_full[p]["mean_mod"]) for p in pids]

    overall_obs_mean = np.nanmean(mean_obs)
    overall_mod_mean = np.nanmean(mean_mod)

    x = np.arange(len(pids))
    width = 0.35

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    axes[0].bar(x, bias, color='tab:red')
    axes[0].set_ylabel('Bias [ppm]')
    axes[0].set_title(f'Bias per site (mean={np.nanmean(bias):.2f})')
    # annotate_bars(axes[0], bias)

    axes[1].bar(x, rmse, alpha=0.8, color='tab:red')
    axes[1].set_ylabel('CRMSE [ppm]')
    axes[1].set_title(f'Centered RMSE per site (mean={np.nanmean(rmse):.2f})')
    # annotate_bars(axes[1], rmse)

    axes[2].bar(x, corr, alpha=0.8, color='tab:red')
    axes[2].set_ylabel('Pearson r')
    axes[2].set_title(f'Correlation per site (mean={np.nanmean(corr):.2f})')
    # annotate_bars(axes[2], corr)

    axes[3].bar(x - width/2, mean_obs, width, label=f'OBS (mean={overall_obs_mean:.2f})', alpha=0.8, color='black')
    axes[3].bar(x + width/2, mean_mod, width, label=f'MOD (mean={overall_mod_mean:.2f})', alpha=0.8, color='red')
    axes[3].set_ylabel('Mean XCO2 [ppm]')
    axes[3].set_title('Mean XCO2 per site')
    axes[3].set_xticks(x)
    # axes[3].set_ylim(bottom=410)
    axes[3].set_xticklabels(pids, rotation=45)
    axes[3].legend()
    # annotate_bars(axes[3], mean_obs)
    # annotate_bars(axes[3], mean_mod)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "metrics_per_site.png"))
    plt.close()


def plot_metrics_month(metrics_month_avg, metrics_month_full, outdir):
    """Plot metrics per month with Bias, CRMSE, Correlation, and Mean OBS/MOD."""
    os.makedirs(outdir, exist_ok=True)
    months = sorted(metrics_month_avg.keys())
    labels = [f"{y}-{m:02d}" for y, m in months]

    bias  = [metrics_month_avg[m]["bias"] for m in months]
    rmse  = [metrics_month_avg[m]["rmse"] for m in months]
    corr  = [metrics_month_avg[m]["corr"] for m in months]
    mean_obs = [np.nanmean(metrics_month_full[m]["mean_obs"]) for m in months]
    mean_mod = [np.nanmean(metrics_month_full[m]["mean_mod"]) for m in months]

    overall_obs_mean = np.nanmean(mean_obs)
    overall_mod_mean = np.nanmean(mean_mod)

    x = np.arange(len(months))
    width = 0.35

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    axes[0].bar(x, bias, alpha=0.8, color='tab:red')
    axes[0].set_ylabel('Bias [ppm]')
    axes[0].set_title(f'Bias per month (mean={np.nanmean(bias):.2f})')
    # annotate_bars(axes[0], bias)

    axes[1].bar(x, rmse, alpha=0.8, color='tab:red')
    axes[1].set_ylabel('CRMSE [ppm]')
    axes[1].set_title(f'Centered RMSE per month (mean={np.nanmean(rmse):.2f})')
    # annotate_bars(axes[1], rmse)

    axes[2].bar(x, corr, alpha=0.8, color='tab:red')
    axes[2].set_ylabel('Pearson r')
    axes[2].set_title(f'Correlation per month (mean={np.nanmean(corr):.2f})')
    # annotate_bars(axes[2], corr)

    axes[3].bar(x - width/2, mean_obs, width, label=f'OBS (mean={overall_obs_mean:.2f})', alpha=0.8, color='black')
    axes[3].bar(x + width/2, mean_mod, width, label=f'MOD (mean={overall_mod_mean:.2f})', alpha=0.8, color='red')
    axes[3].set_ylabel('Mean XCO2 [ppm]')
    axes[3].set_title('Mean XCO2 per site')
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=45)
    # axes[3].set_ylim(bottom=410)  # start y-axis from 410 ppm
    axes[3].legend()
    # annotate_bars(axes[3], mean_obs)
    # annotate_bars(axes[3], mean_mod)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "metrics_per_month.png"))
    plt.close()


# --- Example call ---
metrics_site_avg, metrics_month_avg, metrics_site_full, metrics_month_full = compute_monthly_metrics(ds)
plot_metrics_site(metrics_site_avg, metrics_site_full, outdir=outpath + "Obs_BG/stats/")
plot_metrics_month(metrics_month_avg, metrics_month_full, outdir=outpath + "Obs_BG/stats/")



#----------- DEBUGGING PLOTS ----------

import matplotlib.dates as mdates


def plot_hovmoller_profiles(all_results, pid, savepath="hovmoller_diurnal.png"):
    """
    Plot diurnal cycle (hourly averaged) of vertical XCO2 profiles for total and components.
    """
    # Filter results for this site
    pid_results = [r for r in all_results if r and r['pid'] == pid]
    if not pid_results:
        print(f"No results for {pid}")
        return

    # Extract hours and profiles
    hours = np.array([r['time'].astype('datetime64[h]').astype(int) % 24 for r in pid_results])
    # stack into 2D arrays: time x vertical level
    components = ['cnc_vert_total', 'cnc_vert_Anthr', 'cnc_vert_BG', 'cnc_vert_RA', 'cnc_vert_GPP', 'AVK_avg']
    heights = pid_results[0]['profile_dict']['avk_heights']
    profiles = {comp: np.array([r['profile_dict'][comp] for r in pid_results]) for comp in components}

    # Compute hourly mean for each vertical level
    hourly_profiles = {}
    for comp in components:
        hourly_profiles[comp] = np.zeros((24, len(heights)))
        for h in range(24):
            mask = (hours == h)
            if np.any(mask):
                hourly_profiles[comp][h, :] = np.nanmean(profiles[comp][mask, :], axis=0)
            else:
                hourly_profiles[comp][h, :] = np.nan  # if no data for that hour

    # Create figure
    n_components = len(components)
    fig, axes = plt.subplots(1, n_components, figsize=(4*n_components, 6), sharey=True)
    cmap = 'viridis'
    hour_labels = np.arange(24)
    for ax, comp in zip(axes, components):
        im = ax.pcolormesh(hour_labels, heights[:10], hourly_profiles[comp].T[:10, :], shading='auto', cmap=cmap)
        ax.set_title(comp.replace('cnc_vert_', '').upper())
        ax.set_xlabel('Hour of day')
        if ax == axes[0]:
            ax.set_ylabel('Height above station (m)')
        if comp != 'AVK_avg':
            fig.colorbar(im, ax=ax, label='Interpolated CO2 (ppm)')
        else:
            fig.colorbar(im, ax=ax, label='Averaging Kernel')
    fig.suptitle(f'Diurnal cycle of vertical CO2 profiles for {pid}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(savepath, dpi=150)
    plt.close(fig)




def plot_obs_availability(all_results, pid, savepath="obs_availability.png"):
    """
    Plot number of obs for each hour and whether XCO2_mod / XCO2_obs is valid.
    """
    pid_results = [r for r in all_results if r and r['pid'] == pid]
    if not pid_results:
        print(f"No results for {pid}")
        return
    times = [r['time'] for r in pid_results]
    n_obs = [r.get('n_obs', 0) for r in pid_results]
    obs_valid = [~np.isnan(r['XCO2_obs']) for r in pid_results]
    mod_valid = [~np.isnan(r['XCO2_mod']) for r in pid_results]
    df = pd.DataFrame({
        'n_obs': n_obs,
        'obs_valid': obs_valid,
        'mod_valid': mod_valid
    }, index=pd.to_datetime(times))
    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(df.index, df['n_obs'], width=0.03, label='Number of obs in window', color='lightgray', align='center')
    # Only plot n_obs where obs is valid
    ax.scatter(df.index[df['obs_valid']], df['n_obs'][df['obs_valid']],
    color='green', s=50, label='XCO2_obs valid')


    # Only plot n_obs where model is valid
    ax.scatter(df.index[df['mod_valid']], df['n_obs'][df['mod_valid']],
    color='red', s=50, label='XCO2_mod valid', marker='x')


    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Number of observations')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_title(f'Observation availability for {pid}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)

plot_hovmoller_profiles(all_results, pid="JCMB_SN217", savepath="JCMB_SN217_hovmoller.png")
plot_obs_availability(all_results, pid="JCMB_SN217", savepath="JCMB_SN217_obs_availability.png")


def plot_diurnal_cycle_percent(all_results, pid, savepath_profiles="diurnal_cycle_percent.png",
                               savepath_contrib="diurnal_cycle_contributions.png"):

    # Filter results for this site
    pid_results = [r for r in all_results if r and r['pid'] == pid]
    if not pid_results:
        print(f"No results for {pid}")
        return

    nlayers = 10  # first 10 layers only

    # Extract hours
    hours = np.array([r['time'].astype('datetime64[h]').astype(int) % 24 for r in pid_results])
    heights = pid_results[0]['profile_dict']['avk_heights'][:nlayers]

    # --- 1) Percent deviation per height ---
    components = ['cnc_vert_total', 'cnc_vert_Anthr', 'cnc_vert_BG', 'cnc_vert_RA', 'cnc_vert_GPP', 'AVK_avg']
    component_names = ['Total', 'Anthr', 'BG', 'RA', 'GPP', 'AVK']

    # Stack profiles (first 10 layers)
    profiles = {comp: np.array([r['profile_dict'][comp][:nlayers] for r in pid_results]) for comp in components}

    hour_labels = np.arange(24)

    # Compute percent deviations (skip AVK for % deviation)
    hourly_percent = {}
    for comp in components:
        hourly_mean = np.zeros((24, nlayers))
        for h in range(24):
            mask = (hours == h)
            if np.any(mask):
                hourly_mean[h, :] = np.nanmean(profiles[comp][mask, :], axis=0)
            else:
                hourly_mean[h, :] = np.nan
        
        mean_profile = np.nanmean(hourly_mean, axis=0)
        hourly_percent[comp] = (hourly_mean - mean_profile) / mean_profile * 100

    # --- Plot vertical profiles ---
    fig, axes = plt.subplots(1, len(components), figsize=(25, 5), sharey=True)
    cmap = 'coolwarm'
    for i, comp in enumerate(components):
        # Adjust vmin/vmax
        if comp in ['cnc_vert_total', 'cnc_vert_BG']:
            vmin, vmax = -2, 2
        elif comp == 'AVK_avg':
            vmin, vmax = -10, 10
        else:
            vmin, vmax = -100, 100

        im = axes[i].pcolormesh(hour_labels, heights, hourly_percent[comp].T,
                                shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i].set_title(comp.replace('cnc_vert_', '').replace('_avg', '').upper())
        axes[i].set_xlabel('Hour of day')
        if i == 0:
            axes[i].set_ylabel('Height (m)')

        # Colorbar labeling
        if comp != 'AVK_avg':
            fig.colorbar(im, ax=axes[i], label='% deviation')
        else:
            fig.colorbar(im, ax=axes[i], label='Averaging Kernel')

    fig.suptitle(f'Diurnal % deviation of XCO2 profiles for {pid} (first 10 layers)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(savepath_profiles, dpi=150)
    plt.close(fig)

    # --- 2) Compute surface contributions ---
    component_contribs = ['cnc_vert_Anthr', 'cnc_vert_BG', 'cnc_vert_RA', 'cnc_vert_GPP']
    component_names_contribs = ['Anthr', 'BG', 'RA', 'GPP']
    hourly_unweighted = np.full((24, 4), np.nan)
    hourly_weighted = np.full((24, 4), np.nan)

    for h in range(24):
        mask = (hours == h)
        if np.any(mask):
            contrib_unweighted = []
            contrib_weighted = []
            for r_idx in np.where(mask)[0]:
                r = pid_results[r_idx]
                total_layer = r['profile_dict']['cnc_vert_total']
                # Avoid division by zero
                total_layer_safe = np.where(total_layer == 0, np.nan, total_layer)
                comp_layers = [r['profile_dict'][c] for c in component_contribs]

                # unweighted
                unweighted_vals = [100 * np.nanmean(c / total_layer_safe) for c in comp_layers]
                contrib_unweighted.append(unweighted_vals)

                # AVK-weighted
                AVK = r['profile_dict']['AVK_avg']
                weighted_vals = [100 * np.nansum(c * AVK) / np.nansum(total_layer_safe * AVK) for c in comp_layers]
                contrib_weighted.append(weighted_vals)

            if contrib_unweighted:
                hourly_unweighted[h, :] = np.nanmean(contrib_unweighted, axis=0)
            if contrib_weighted:
                hourly_weighted[h, :] = np.nanmean(contrib_weighted, axis=0)

    # --- Plot contributions ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange']

    # Subplot 1: unweighted (solid) and AVK-weighted (dashed)
    for i in range(4):
        mean_unw = np.nanmean(hourly_unweighted[:, i])
        mean_w = np.nanmean(hourly_weighted[:, i])
        axes[0].plot(hour_labels, hourly_unweighted[:, i], label=f"{component_names[i]} (mean {mean_unw:.1f}%)", 
                     color=colors[i], linewidth=2)
        axes[0].plot(hour_labels, hourly_weighted[:, i], linestyle='--', color=colors[i], linewidth=2)
    axes[0].set_ylabel('Contribution (%)')
    axes[0].set_title(f'Unweighted & AVK-weighted contributions (first 10 layers) for {pid}')
    axes[0].legend()
    axes[0].grid(True)
    # Optionally, logarithmic scale if needed:
    # axes[0].set_yscale('log')

    # Subplot 2: difference weighted - unweighted
    for i in range(4):
        diff = hourly_weighted[:, i] - hourly_unweighted[:, i]
        axes[1].plot(hour_labels, diff, label=component_names[i], color=colors[i], linewidth=2)
    axes[1].set_ylabel('Difference in contribution (%)')
    axes[1].set_xlabel('Hour of day')
    axes[1].set_title('Difference between AVK-weighted and unweighted contributions')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(savepath_contrib, dpi=150)
    plt.close(fig)

# Example usage:
plot_diurnal_cycle_percent(all_results, pid="JCMB_SN217",
                            savepath_profiles="JCMB_SN217_DC_profiles.png",
                            savepath_contrib="JCMB_SN217_DC_contributions.png")