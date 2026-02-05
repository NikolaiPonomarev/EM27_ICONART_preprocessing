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
default_obs_version = '*V015*'
def_v1b = "V008"
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

def get_first_valid_station_per_pid(file_pairs):
    """
    Return one valid Station object per PID.
    """
    pid_to_station = OrderedDict()
    pid_files = defaultdict(list)

    # Group obs files by PID
    for _, obs_file in file_pairs:
        pid = get_pid_from_filename(obs_file)
        if pid is not None:
            pid_files[pid].append(obs_file)

    # Try each obs file until a valid one is found
    for pid, files in pid_files.items():
        for f in files:
            s = Station(f, lons_km, lats_km)
            if s.valid:
                pid_to_station[pid] = s
                break
        else:
            print(f"WARNING: No valid obs file found for PID {pid}")

    return pid_to_station

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




file_pairs = get_file_pairs(model_dir, obs_dir, obs_v=default_obs_version)

# Precompute station info from the first obs file
stations = get_first_valid_station_per_pid(file_pairs)


def linear_extrapolate(avk_heights, z_model, cnc_model, cnc_prior, is_component=False):
    """
    Interpolate CO2 profile to retrieval heights:
      - Bottom, Below model layer heights: constant (lowest model layer)
      - Mid, In range with model layer heights: linear interpolation
      - Top, Above model layer heights: use prior profile
    """
    y = np.empty_like(avk_heights)

    # Bottom: constant
    mask_bottom = avk_heights < z_model[0]
    y[mask_bottom] = cnc_model[0]

    # Middle: linear interpolation (between model min and max)
    mask_middle = (avk_heights >= z_model[0]) & (avk_heights <= z_model[-1])
    y[mask_middle] = np.interp(avk_heights[mask_middle], z_model, cnc_model)

    # Top: use prior
    mask_top = avk_heights > z_model[-1]
    if is_component:#linearly extrapolate component profiles, their values are only relevant near the surface
        slope_top = (cnc_model[-1] - cnc_model[-2]) / (z_model[-1] - z_model[-2])
        y[mask_top] = cnc_model[-1] + slope_top * (avk_heights[mask_top] - z_model[-1])
    else:#for total profile use prior at the top
        y[mask_top] = cnc_prior[mask_top]

    return y


def interp_rho_exp(avk_heights, z_model, rho_model, Nfit=5):
    """
    Interpolate density profile to retrieval heights:
      - Bottom: constant (lowest model layer)
      - Middle: linear interpolation
      - Top: exponential fit to top Nfit layers
    """
    y = np.empty_like(avk_heights)

    # Bottom: constant
    mask_bottom = avk_heights < z_model[0]
    if np.any(mask_bottom):
        y[mask_bottom] = rho_model[0]

    # Middle: linear interpolation
    mask_middle = (avk_heights >= z_model[0]) & (avk_heights <= z_model[-1])
    y[mask_middle] = np.interp(avk_heights[mask_middle], z_model, rho_model)

    # Top: exponential fit
    mask_top = avk_heights > z_model[-1]
    def rho_exp(z, rho0, H):
            return rho0 * np.exp(-(z - z_fit[0]) / H)
    if np.any(mask_top):
        z_fit = z_model[-Nfit:]
        rho_fit = rho_model[-Nfit:]
        params, _ = curve_fit(rho_exp, z_fit, rho_fit, p0=[rho_fit[0], 8000.0])
        y[mask_top] = rho_exp(avk_heights[mask_top], *params)

    return y


def debug_plot_stepwise(
        z_model, cnc_model,
        z_interp, cnc_interp,
        cnc_prior,
        avk, pid, timestamp,
        cnc_model_comp,     
        cnc_interp_comp,     
        comp_name,
        x_retr_comp,
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

    axs[0,1].plot(cnc_interp, z_interp, 'b.-', label="Model interp")
    axs[0,1].plot(cnc_prior,  z_interp, 'g.--', label="Prior")
    axs[0,1].set_title("Interpolated ICON-ART(TOTAL) and Prior")
    axs[0,1].set_xlabel("CO₂ (ppm)")
    axs[0,1].set_ylabel("z (m)")

    axs[0,2].plot(avk, z_interp, 'r.-')
    axs[0,2].set_title("Averaging Kernel")
    axs[0,2].set_xlabel("AK")
    axs[0,2].set_ylabel("z (m)")

    contrib_total = cnc_prior + avk * (cnc_interp - cnc_prior)
    axs[0,3].scatter(contrib_total, z_interp, s=40)
    axs[0,3].set_title("Contribution (TOTAL)")
    axs[0,3].set_xlabel("x̂ = xa + A(x-xa)")
    axs[0,3].legend([
        f"XCO₂_total_model = Σa.m.(cnc_pr + AK(cnc_int - cnc_pr)) / Σ(a.m.)"
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

    
    axs[1,3].plot(x_retr_comp, z_interp, 'm.-')
    axs[1,3].set_title(f"Retrieved Profile ({comp_name})")
    axs[1,3].set_xlabel("Retrieved CO₂ (ppm)")
    axs[1,3].legend([
        f"XCO₂_({comp_name}) = {XCO2_mod_comp:.2f} ppm"
    ], loc="upper right")

    fig.suptitle(f"{pid} – {timestamp} – Component {comp_name}")
    plt.tight_layout()

    fname = os.path.join(outdir, f"debug_{pid}_{timestamp}_{comp_name}.png")
    plt.savefig(fname)
    plt.close()
    print(fname)

def debug_plot_rho(z_model, rho_model, z_interp, rho_interp, pid, timestamp, outdir="debug_plots"):
    """
    Debug plot for density profile: model vs interpolated/extrapolated
    """
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.plot(rho_model, z_model, 'k.-', label="Model rho")
    ax.plot(rho_interp, z_interp, 'b.-', label="Interpolated/extrapolated rho")
    ax.set_xlabel("Density (kg/m³)")
    ax.set_ylabel("Height (m)")
    ax.set_title(f"{pid} – {timestamp} – Density Profile")
    ax.legend()
    plt.tight_layout()
    fname = os.path.join(outdir, f"debug_rho_{pid}_{timestamp}.png")
    plt.savefig(fname)
    plt.close()
    print(fname)

def l2_to_l1_filename(l2_path, l1_version=def_v1b):
    """
    Convert L2 EM27 filename to corresponding L1b filename.
    """

    # Replace directory
    l1_path = l2_path.replace("/L2/", "/L1/")

    # Replace file tag
    l1_path = l1_path.replace("_L2_", "_L1b_")

    # Replace version
    l1_path = re.sub(r"_V\d{3}\.nc$", f"_{l1_version}.nc", l1_path)

    return l1_path


def apply_prior_ak_mass(cnc_vert_model, prior_profile, AK, rho, dz):

    x_retr = prior_profile + AK * (cnc_vert_model - prior_profile)
    w = rho * dz
    return np.nansum(x_retr * w) / np.nansum(w)
def apply_prior_ak_mass_component(cnc_vert_model, prior_profile, AK, fraction_prof, rho, dz):

    x_retr = fraction_prof * (prior_profile + AK * (cnc_vert_model - prior_profile))
    w = rho * dz
    return np.nansum(x_retr * w) / np.nansum(w)


def process_file_pair_full(file_pair, stations, min_nobs=7):
    
    model_file, obs_file = file_pair
    
    pid = get_pid_from_filename(obs_file)
    if pid not in stations:
        print(f"Skipping {pid}: no valid Station info")
        return ("fail_no_station", {"pid": pid, "model_file": model_file, "obs_file": obs_file})
    station = stations[pid]
    print('Processing', model_file, obs_file, pid)
    # Open datasets
    ds_mod = xr.open_dataset(model_file)
    ds_obs = xr.open_dataset(obs_file)
    ds_obsl1 = xr.open_dataset(l2_to_l1_filename(obs_file, l1_version="V008"))

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
    n_valid_obs = obs_in_window.XCO2.count(dim='time').values
    if n_valid_obs<min_nobs:
        return ("fail_too_few_obs", {"pid": pid, "model_file": model_file, "obs_file": obs_file})
    else:
        XCO2_obs_avg = obs_in_window.XCO2.mean(dim='time', skipna=True).values
        XCO2_prior = ds_obsl1.CO2_prior.values[0, :] * 1e6 # midday prior profile from L1b data, MAP files contain 3hr average
    AVK_avg = obs_in_window.XCO2_AK.mean(dim='time', skipna=True).values
    if np.isnan(station.topo_obs):
            return ("fail_nan_topo", {"pid": pid, "model_file": model_file, "obs_file": obs_file})
    if np.any(np.isnan(AVK_avg)):
            return ("fail_nan_avk", {"pid": pid, "model_file": model_file, "obs_file": obs_file})
    if np.any(np.isnan(XCO2_prior)):
            return ("fail_nan_prior", {"pid": pid, "model_file": model_file, "obs_file": obs_file})

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
    rho_hor = hor_interp(ds_mod.rho[0].values, station)
    print('Model topo', mod_topo_hor[-1], 'Observation topo ', station.topo_obs)

    # ---------- Vertical interpolation ----------
    avk_heights_total = station.avk_heights_m  - station.topo_obs#+ station.height_agl
    if np.any(np.isnan(avk_heights_total)):
            return ("fail_nan_heights", {"pid": pid, "model_file": model_file, "obs_file": obs_file})

    # ---------- no extrapolation ----------
    cnc_vert = linear_extrapolate(avk_heights_total, z_ifc_hor[::-1], cnc_hor_total[::-1], XCO2_prior)
    cnc_vert_TR = linear_extrapolate(avk_heights_total, z_ifc_hor[::-1], cnc_hor_TR[::-1], XCO2_prior, is_component=True)
    #cnc_vert_BG = linear_extrapolate(avk_heights_total, z_ifc_hor[::-1], cnc_hor_TR_BG[::-1], XCO2_prior)
    cnc_vert_RA = linear_extrapolate(avk_heights_total, z_ifc_hor[::-1], cnc_hor_RA[::-1], XCO2_prior, is_component=True)
    cnc_vert_GPP = linear_extrapolate(avk_heights_total, z_ifc_hor[::-1], cnc_hor_GPP[::-1], XCO2_prior, is_component=True)
    cnc_vert_BG = cnc_vert - cnc_vert_TR - cnc_vert_RA + cnc_vert_GPP
    rho_vert = interp_rho_exp(avk_heights_total, z_ifc_hor[::-1], rho_hor[::-1])

    dz = np.diff(avk_heights_total, prepend=0.0)  # prepend surface height
    # print('DZ values:', dz, 'avkh', avk_heights_total)
    # ---------- OCO-2 style retrieval with prior ----------
    XCO2_mod = apply_prior_ak_mass(cnc_vert, XCO2_prior, AVK_avg, rho_vert, dz)

    XCO2_mod_TR  = apply_prior_ak_mass_component(cnc_vert, XCO2_prior, AVK_avg, cnc_vert_TR  / cnc_vert,  rho_vert, dz)
    XCO2_mod_BG  = apply_prior_ak_mass_component(cnc_vert, XCO2_prior, AVK_avg, cnc_vert_BG  / cnc_vert,  rho_vert, dz)
    XCO2_mod_RA  = apply_prior_ak_mass_component(cnc_vert, XCO2_prior, AVK_avg, cnc_vert_RA  / cnc_vert,  rho_vert, dz)
    XCO2_mod_GPP = apply_prior_ak_mass_component(cnc_vert, XCO2_prior, AVK_avg, cnc_vert_GPP  / cnc_vert, rho_vert, dz)

    if np.isfinite(XCO2_mod) and XCO2_mod == 0.0:

        STATS["fail_zero_xco2"] += 1
        DEBUG_FAIL["zero_xco2"].append((pid, model_file, obs_file))

        print("\n=== ZERO XCO2 DETECTED ===")
        print("PID:", pid)
        print("Model:", model_file)
        print("Obs:", obs_file)
        print("Time:", model_time)
        print("AVK sum:", np.nansum(AVK_avg))
        print("w sum:", np.nansum(rho_vert * dz))
        print("dz min:", np.min(dz))
        print("rho nan:", np.isnan(rho_vert).sum())
        print("cnc min:", np.min(cnc_vert))
        print("========================\n")
        return ("fail_zero_xco2", {"pid": pid, "model_file": model_file, "obs_file": obs_file})
    # Close datasets
    ds_mod.close()
    ds_obs.close()
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
        'obs_topo': station.topo_obs
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
all_results_valid = [d for d in all_results if isinstance(d, dict) and 'time' in d and 'pid' in d]
# Collect all failed results
all_results_failed = [d for d in all_results if isinstance(d, tuple)]

# Group by failure reason
fail_summary = defaultdict(list)
for fail in all_results_failed:
    reason = fail[0]
    info = fail[1]
    fail_summary[reason].append(info)

# Now printing works:
print("=== Failed case summary ===")
for reason, entries in fail_summary.items():
    print(f"Due to {reason}: {len(entries)} failures")
    for e in entries[:3]:
        print("E.g. PID:", e['pid'], "| Model:", e['model_file'], "| Obs:", e['obs_file'])

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






def plot_hourly_month(ds, pid, year, month, outdir=outpath + "Mod_Prior"):
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
    plt.plot(times, obs, 'k-o', lw=1.5, markersize=3.5, mfc='none', label=f"OBS ({stats_dict['OBS']})")
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
    n_obs_site = {}
    n_obs_month = {}

    all_times = pd.to_datetime(ds.time.values)
    months = sorted(set((t.year, t.month) for t in all_times))

    for pid in ds.pid.values:
        metrics_site[pid] = {"bias": [], "rmse": [], "corr": [], "mean": []}
        metrics_site_full[pid] = {"mean_obs": [], "mean_mod": []}
        n_obs_site[pid] = []
    for year, month in months:
        for pid in ds.pid.values:
            sel = ds.sel(pid=pid)
            time_sel = (sel.time.dt.year == year) & (sel.time.dt.month == month)
            d = sel.sel(time=time_sel)
            obs = d["XCO2_obs"].values #- d['XCO2_mod_BG'].values
            mod = d["XCO2_mod"].values

            mask = ~np.isnan(obs) & ~np.isnan(mod)
            n_valid = np.sum(mask)
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
            n_obs_site[pid].append(n_valid)
            # Save per month
            metrics_month.setdefault((year, month), {"bias": [], "rmse": [], "corr": [], "mean": []})
            metrics_month_full.setdefault((year, month), {"mean_obs": [], "mean_mod": []})

            metrics_month[(year, month)]["bias"].append(bias)
            metrics_month[(year, month)]["rmse"].append(crmse)
            metrics_month[(year, month)]["corr"].append(corr)
            metrics_month[(year, month)]["mean"].append(mean_obs)
            n_obs_month.setdefault((year, month), []).append(n_valid)
            metrics_month_full[(year, month)]["mean_obs"].append(mean_obs)
            metrics_month_full[(year, month)]["mean_mod"].append(mean_mod)

    # Average metrics over months/sites
    metrics_site_avg = {pid: {k: np.nanmean(v) for k, v in metrics_site[pid].items()} for pid in metrics_site}
    metrics_month_avg = {m: {k: np.nanmean(v) for k, v in metrics_month[m].items()} for m in metrics_month}

    return metrics_site_avg, metrics_month_avg, metrics_site_full, metrics_month_full, n_obs_site, n_obs_month


def annotate_bars(ax, values):
    for i, v in enumerate(values):
        ax.text(i, v + 0.01 * max(values), f"{v:.2f}", ha='center', va='bottom', fontsize=9)


def plot_metrics_site(metrics_site_avg, metrics_site_full, n_obs_site, outdir):
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

    axes[3].bar(x - width/2, mean_obs, width, label=f'OBS (mean={overall_obs_mean:.3f})', alpha=0.8, color='black')
    axes[3].bar(x + width/2, mean_mod, width, label=f'MOD (mean={overall_mod_mean:.3f})', alpha=0.8, color='red')
    axes[3].set_ylabel('Mean XCO2 [ppm]')
    axes[3].set_title('Mean XCO2 per site')
    axes[3].set_xticks(x)
    for i, pid in enumerate(pids):
        n_obs = sum(n_obs_site[pid])
        axes[3].text(i - width/2, mean_obs[i] + 0.5, f"{n_obs}", ha='center', va='bottom', fontsize=9, color='blue')
    axes[3].set_ylim(bottom=415)
    axes[3].set_xticklabels(pids, rotation=45)
    axes[3].legend()
    # annotate_bars(axes[3], mean_obs)
    # annotate_bars(axes[3], mean_mod)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "metrics_per_site.png"))
    plt.close()


def plot_metrics_month(metrics_month_avg, metrics_month_full, n_obs_month, outdir):
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
    print('Obs:', overall_obs_mean, 'Mod:', overall_mod_mean)
    axes[3].bar(x - width/2, mean_obs, width, label=f'OBS (mean={overall_obs_mean:.3f})', alpha=0.8, color='black')
    axes[3].bar(x + width/2, mean_mod, width, label=f'MOD (mean={overall_mod_mean:.3f})', alpha=0.8, color='red')
    axes[3].set_ylabel('Mean XCO2 [ppm]')
    axes[3].set_title('Mean XCO2 per site')
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=45)
    for i, m in enumerate(months):
        n_obs = sum(n_obs_month[m])
        axes[3].text(i - width/2, mean_obs[i] + 0.5, f"{n_obs}", ha='center', va='bottom', fontsize=9, color='blue')
    axes[3].set_ylim(bottom=415)  # start y-axis from 410 ppm
    axes[3].legend()
    # annotate_bars(axes[3], mean_obs)
    # annotate_bars(axes[3], mean_mod)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "metrics_per_month.png"))
    plt.close()


# --- Example call ---
metrics_site_avg, metrics_month_avg, metrics_site_full, metrics_month_full, n_obs_site, n_obs_month = compute_monthly_metrics(ds)
plot_metrics_site(metrics_site_avg, metrics_site_full, n_obs_site, outdir=outpath + "Mod_Prior/stats/")
plot_metrics_month(metrics_month_avg, metrics_month_full, n_obs_month, outdir=outpath + "Mod_Prior/stats/")