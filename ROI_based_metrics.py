"""
ROI-Based Spatial Leakage Mapping.

This script extracts and maps the spatial leakage of the Point-Spread Function (PSF)
and Cross-Talk Function (CTF) for specific visual processing functional Regions
of Interest (ROIs).

The ROIs are defined using the HCP-MMP1 parcellation on the 'fsaverage' template:
- V1 (Primary Visual Cortex)
- Face Network (FFC, PIT)
- Motion Network (MT, MST)

It computes the grand average spread across subjects, normalizes the maps, and
generates a comprehensive grid visualization comparing the four inverse methods.
"""

import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mne.minimum_norm import make_inverse_resolution_matrix, get_point_spread, get_cross_talk

# Setup directories
derivatives_dir = r"ds003505\derivatives"
subjects_dir = os.path.join(derivatives_dir, 'subjects')
out_dir = os.path.join(derivatives_dir, 'resolution_metrics', 'roi_maps')
os.makedirs(out_dir, exist_ok=True)

# List of subjects
sub_ids = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-06', 'sub-07', 'sub-08',
    'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15',
    'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'
]

SPACING = 'ico5' # Corrected from 'isco5' to match other files
methods = ['MNE', 'dSPM', 'sLORETA', 'eLORETA']
snr = 3.0
lambda2 = 1.0 / snr ** 2

# Configuration defining the target functional ROIs and optimal viewing angles
roi_config = {
    'V1': {'labels': ['V1'], 'view': 'caudal'},
    'Face': {'labels': ['FFC', 'PIT'], 'view': 'ventral'},
    'Motion': {'labels': ['MT', 'MST'], 'view': 'lateral'}
}

# Fetch the HCP-MMP1 parcellation scheme (Glasser et al., 2016)
mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, verbose=False)
all_labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', subjects_dir=subjects_dir)

# Construct target labels by merging relevant HCP regions
target_labels = {}
for net_name, config in roi_config.items():
    parts = config['labels']
    # Filter for parts on the left hemisphere (we use lh for visualization)
    lh_parts = [l for l in all_labels if any(f'_{p}_' in l.name for p in parts) and l.hemi == 'lh']

    if lh_parts:
        merged = lh_parts[0].copy()
        for l in lh_parts[1:]:
            merged += l
        merged.name = net_name
        target_labels[net_name] = merged

# Initialize dictionary to hold Grand Average maps
ga_maps = {m: {r: {'psf': None, 'ctf': None} for r in target_labels} for m in methods}
n_subjects = len(sub_ids)

# Process spatial spread for each subject
for sub_id in sub_ids:
    print(f"Processing ROI leakage for {sub_id}...")
    fwd_fname = os.path.join(derivatives_dir, 'forward_solutions', f"{sub_id}_{SPACING}-fwd.fif")
    inv_fname = os.path.join(derivatives_dir, 'inverse_operators', f"{sub_id}_{SPACING}-inv.fif")

    try:
        fwd = mne.read_forward_solution(fwd_fname, verbose=False)
        fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=False)
        inv_op = mne.minimum_norm.read_inverse_operator(inv_fname, verbose=False)
        src = inv_op['src']

        for method in methods:
            # Reconstruct resolution matrix
            resmat = make_inverse_resolution_matrix(fwd, inv_op, method=method, lambda2=lambda2, verbose=False)

            for roi_name, label in target_labels.items():
                # Get mean spatial spread (PSF) originating from the ROI
                psf_out = get_point_spread(resmat, src, [label], mode='mean', verbose=False)
                stc_psf = psf_out[0] if isinstance(psf_out, list) else psf_out

                # Get mean cross-talk (CTF) bleeding into the ROI
                ctf_out = get_cross_talk(resmat, src, [label], mode='mean', verbose=False)
                stc_ctf = ctf_out[0] if isinstance(ctf_out, list) else ctf_out

                # Normalize maps to maximum value of 1 for better visual comparison of spread
                if np.max(np.abs(stc_psf.data)) > 0: stc_psf.data /= np.max(np.abs(stc_psf.data))
                if np.max(np.abs(stc_ctf.data)) > 0: stc_ctf.data /= np.max(np.abs(stc_ctf.data))

                # Accumulate for Grand Average
                if ga_maps[method][roi_name]['psf'] is None:
                    ga_maps[method][roi_name]['psf'] = stc_psf
                    ga_maps[method][roi_name]['ctf'] = stc_ctf
                else:
                    ga_maps[method][roi_name]['psf'].data += stc_psf.data
                    ga_maps[method][roi_name]['ctf'].data += stc_ctf.data

            del resmat

    except Exception as e:
        print(f"  Warning: Skipping {sub_id} due to error: {e}")

# Compute average across all subjects
for m in methods:
    for r in target_labels:
        if ga_maps[m][r]['psf'] is not None:
            ga_maps[m][r]['psf'].data /= n_subjects
            ga_maps[m][r]['ctf'].data /= n_subjects

# Attempt to set optimal 3D backend
try:
    mne.viz.set_3d_backend('pyvistaqt')
except Exception:
    pass

# Helper to remove whitespace from brain screenshots
def crop_image(img_array):
    # Find pixels that are not pure white
    is_data = np.any(img_array < 250, axis=2)
    coords = np.argwhere(is_data)
    if coords.size == 0: return img_array
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return img_array[y_min:y_max+1, x_min:x_max+1, :]

# Setup visualization grid
print("Generating ROI spread visualizations...")
roi_order = ['V1', 'Face', 'Motion']
col_types = ['psf', 'ctf']

fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(16, 9),
                         gridspec_kw={'wspace': 0.05, 'hspace': 0.05, 'left': 0.08, 'right': 0.92})

# Plot each map into the grid
for i, method in enumerate(methods):
    col_idx = 0
    for roi_name in roi_order:
        view_angle = roi_config[roi_name]['view']

        for func_type in col_types:
            ax = axes[i, col_idx]
            stc = ga_maps[method][roi_name][func_type]

            if stc is not None:
                # Plot onto fsaverage inflated surface
                brain = stc.plot(
                    subject='fsaverage', subjects_dir=subjects_dir, hemi='lh',
                    views=view_angle, surface='inflated',
                    colormap='hot', clim=dict(kind='value', lims=[0, 0.5, 1.0]),
                    colorbar=False, time_viewer=False, show_traces=False,
                    background='white', size=(400, 300)
                )

                # Draw the outline of the ROI in blue
                brain.add_label(target_labels[roi_name], borders=True, color='blue')
                img = brain.screenshot()
                ax.imshow(crop_image(img))
                brain.close()

            ax.axis('off')

            # Add column headers
            if i == 0:
                ax.set_title(f"{roi_name}\n{func_type.upper()}", fontsize=12, fontweight='bold')

            # Add row labels
            if col_idx == 0:
                ax.text(-0.3, 0.5, method, transform=ax.transAxes, rotation=90,
                        va='center', ha='center', fontsize=14, fontweight='bold')

            col_idx += 1

# Add global colorbar
cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
norm = mpl.colors.Normalize(vmin=0, vmax=1)
cmap = plt.get_cmap('hot')
mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

cbar = fig.colorbar(mappable, cax=cbar_ax)
cbar.set_label('Normalized Spread (A.U.)')
cbar.ax.tick_params(labelsize=10)

# Save the final composite figure
out_file = os.path.join(out_dir, f'ROIs_Leakage_{SPACING}.png')
plt.savefig(out_file, dpi=600, bbox_inches='tight')
print(f"Saved visualization to: {out_file}")
