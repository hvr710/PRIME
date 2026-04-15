import h5py
import numpy as np
from scipy.signal import welch
from scipy.integrate import simpson
from scipy.stats import entropy
from pathlib import Path

# Configuration
DATA_DIR = Path(__file__).resolve().parent
input_file = DATA_DIR / 'comp4_len5_step5_mapped60.h5'
output_file = DATA_DIR / 'features_comp4_len5_step5_mapped60.h5'
fs = 125  # Sampling rate: 625 samples / 5 seconds
META_DATASETS = [
    'label',
    'split',
    'subject_index',
    'trial_index',
    'segment_index',
    'global_trial_index',
    'segment_start_sample',
    'segment_start_second',
]

# Frequency bands
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

def get_band_power(psd, freqs, band):
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    # Use Simpson's rule to integrate
    return simpson(psd[:, :, idx_band], freqs[idx_band], axis=-1)

def get_hjorth_params(data):
    """Calculate Hjorth Activity, Mobility, and Complexity."""
    # data shape: (segments, channels, samples)
    # Activity = variance
    activity = np.var(data, axis=-1)
    
    # First derivative
    diff1 = np.diff(data, axis=-1)
    var_diff1 = np.var(diff1, axis=-1)
    mobility = np.sqrt(var_diff1 / (activity + 1e-10))
    
    # Second derivative
    diff2 = np.diff(diff1, axis=-1)
    var_diff2 = np.var(diff2, axis=-1)
    mobility_diff1 = np.sqrt(var_diff2 / (var_diff1 + 1e-10))
    complexity = mobility_diff1 / (mobility + 1e-10)
    
    return activity, mobility, complexity

def get_petrosian_fd(data):
    """Petrosian Fractal Dimension."""
    # data shape: (segments, channels, samples)
    n = data.shape[-1]
    # Number of sign changes in the first derivative
    diff = np.diff(data, axis=-1)
    n_delta = np.sum(diff[:, :, 1:] * diff[:, :, :-1] < 0, axis=-1)
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_delta + 1e-10)))

def extract_features():
    with h5py.File(input_file, 'r') as f:
        eeg_data = f['eeg'][:]  # (segments, channels, samples)
        channel_names = [c.decode() if isinstance(c, bytes) else c for c in f['meta/mapped_channel_names']]
        copied_metadata = {name: f[name][:] for name in META_DATASETS if name in f}
        copied_attrs = {
            'metadata_source_h5': str(input_file),
            'metadata_attached': True,
            'split_codebook': f.attrs.get('split_codebook', '{"train": 0, "val": 1, "test": 2}'),
            'fs': f.attrs.get('fs', fs),
            'segment_seconds': f.attrs.get('segment_seconds', 5),
            'segment_step_seconds': f.attrs.get('segment_step_seconds', 5),
            'segment_points': f.attrs.get('segment_points', eeg_data.shape[-1]),
        }
        copied_meta_arrays = {}
        for src_name, dst_name in [
            ('meta/subject_names', 'subject_names'),
            ('meta/subject_groups', 'subject_groups'),
            ('meta/split_names', 'split_names'),
        ]:
            if src_name in f:
                copied_meta_arrays[dst_name] = f[src_name][:]
        
    n_segments, n_channels, n_samples = eeg_data.shape
    print(f"Processing {n_segments} segments, {n_channels} channels, {n_samples} samples per segment...")

    # Calculate PSD for all segments and channels
    # Using window size of 125 (1s) with 50% overlap for Welch
    freqs, psd = welch(eeg_data, fs=fs, nperseg=125, noverlap=62, axis=-1)
    
    # 1. Absolute Power (AP)
    ap_features = {}
    for band_name, band_range in BANDS.items():
        ap_features[band_name] = get_band_power(psd, freqs, band_range)
    
    # 2. Relative Power (RP)
    # Total power (1-45 Hz)
    total_power = get_band_power(psd, freqs, (1, 45))
    rp_features = {}
    for band_name, ap in ap_features.items():
        rp_features[band_name] = ap / (total_power + 1e-10)
        
    # 3. DE / log power
    de_features = {}
    for band_name, ap in ap_features.items():
        de_features[band_name] = 0.5 * np.log(2 * np.pi * np.e * ap + 1e-10) # Approximation of DE for Gaussian

    # 4. Asymmetry
    # Map symmetric channels
    symmetric_pairs = []
    for i, name1 in enumerate(channel_names):
        if name1.endswith('Z'): continue # Skip midline
        # Find matching pair
        if '1' in name1: pair_name = name1.replace('1', '2')
        elif '2' in name1: continue # Handled by '1'
        elif '3' in name1: pair_name = name1.replace('3', '4')
        elif '4' in name1: continue # Handled by '3'
        elif '5' in name1: pair_name = name1.replace('5', '6')
        elif '6' in name1: continue
        elif '7' in name1: pair_name = name1.replace('7', '8')
        elif '8' in name1: continue
        else: continue
        
        if pair_name in channel_names:
            j = channel_names.index(pair_name)
            symmetric_pairs.append((i, j, name1, pair_name))
            
    asym_features = {}
    for band_name, ap in ap_features.items():
        asym_vals = []
        asym_names = []
        for i, j, name_l, name_r in symmetric_pairs:
            # Asymmetry = (AP_L - AP_R) / (AP_L + AP_R) or log(AP_L) - log(AP_R)
            val = np.log(ap[:, i] + 1e-10) - np.log(ap[:, j] + 1e-10)
            asym_vals.append(val)
            asym_names.append(f"{name_l}_{name_r}")
        asym_features[band_name] = np.stack(asym_vals, axis=1)
        asym_features[f"{band_name}_names"] = asym_names

    # 5. Theta/Beta Ratio
    tbr = ap_features['theta'] / (ap_features['beta'] + 1e-10)
    
    # 6. Alpha/Beta Ratio
    abr = ap_features['alpha'] / (ap_features['beta'] + 1e-10)
    
    # 7. Frontal Alpha Asymmetry (FAA)
    frontal_keywords = ['FP', 'AF', 'F']
    faa_indices = [idx for idx, (i, j, nl, nr) in enumerate(symmetric_pairs) 
                   if any(nl.startswith(kw) for kw in frontal_keywords)]
    faa = -asym_features['alpha'][:, faa_indices] # - because asym was L-R, FAA is R-L
    faa_names = [asym_features['alpha_names'][idx] for idx in faa_indices]

    # --- New Emotion-related Features ---
    # 8. Hjorth Parameters
    hj_activity, hj_mobility, hj_complexity = get_hjorth_params(eeg_data)
    
    # 9. Spectral Entropy (normalized)
    psd_norm = psd / (np.sum(psd, axis=-1, keepdims=True) + 1e-10)
    se = entropy(psd_norm, axis=-1) / np.log(psd.shape[-1])
    
    # 10. Petrosian Fractal Dimension
    pfd = get_petrosian_fd(eeg_data)
    
    # 11. Standard Deviation
    std = np.std(eeg_data, axis=-1)

    # Save features to HDF5
    with h5py.File(output_file, 'w') as f:
        # Save absolute power
        ap_grp = f.create_group('absolute_power')
        for b, data in ap_features.items():
            ap_grp.create_dataset(b, data=data)
        
        # Save relative power
        rp_grp = f.create_group('relative_power')
        for b, data in rp_features.items():
            rp_grp.create_dataset(b, data=data)
            
        # Save DE/log power
        de_grp = f.create_group('de_log_power')
        for b, data in de_features.items():
            de_grp.create_dataset(b, data=data)
            
        # Save Asymmetry
        asym_grp = f.create_group('asymmetry')
        for b, data in asym_features.items():
            if isinstance(data, list): # Names
                f.create_dataset(f'asymmetry/{b}', data=np.array(data, dtype='S'))
            else:
                asym_grp.create_dataset(b, data=data)
        
        # Save Ratios
        f.create_dataset('ratios/theta_beta', data=tbr)
        f.create_dataset('ratios/alpha_beta', data=abr)
        
        # Save FAA
        f.create_dataset('faa/values', data=faa)
        f.create_dataset('faa/names', data=np.array(faa_names, dtype='S'))

        # --- Save New Features ---
        hj_grp = f.create_group('hjorth')
        hj_grp.create_dataset('activity', data=hj_activity)
        hj_grp.create_dataset('mobility', data=hj_mobility)
        hj_grp.create_dataset('complexity', data=hj_complexity)
        
        f.create_dataset('spectral_entropy', data=se)
        f.create_dataset('pfd', data=pfd)
        f.create_dataset('std', data=std)
        
        # Save metadata
        f.create_dataset('meta/channel_names', data=np.array(channel_names, dtype='S'))
        f.create_dataset('meta/bands', data=np.array(list(BANDS.keys()), dtype='S'))
        for name, data in copied_metadata.items():
            f.create_dataset(name, data=data)
        for name, value in copied_attrs.items():
            f.attrs[name] = value
        for name, data in copied_meta_arrays.items():
            f.create_dataset(f'meta/{name}', data=data)

    print(f"Features extracted and saved to {output_file}")

if __name__ == "__main__":
    extract_features()
