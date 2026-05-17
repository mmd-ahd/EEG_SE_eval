"""
Resolution Metrics Computation.

This script computes theoretical resolution metrics based on the resolution
matrix for four inverse solutions: MNE, dSPM, sLORETA, and eLORETA.

It calculates:
1. Peak Localization Error (PLE)
2. Spatial Deviation (SD)

These metrics are derived for both the Point-Spread Function (PSF) and the
Cross-Talk Function (CTF) for all specified source space spacings. The
results are saved as MNE SourceEstimates (.stc files).
"""

import os
import mne
from mne.minimum_norm import make_inverse_resolution_matrix, resolution_metrics

# Define directories for inputs and outputs
derivatives_dir = r"ds003505\derivatives"
fwd_dir = os.path.join(derivatives_dir, 'forward_solutions')
inv_dir = os.path.join(derivatives_dir, 'inverse_operators')
subjects_dir = os.path.join(derivatives_dir, 'subjects')

out_dir = os.path.join(derivatives_dir, 'resolution_metrics')
os.makedirs(out_dir, exist_ok=True)

# List of subject IDs to process
sub_ids = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-06', 'sub-07', 'sub-08',
    'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15',
    'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'
]

# Experimental parameters
spacings = ['ico4', 'ico5', 'oct6']
methods = ['MNE', 'dSPM', 'sLORETA', 'eLORETA']
snr = 3.0
lambda2 = 1.0 / snr ** 2  # Regularization parameter

for spacing in spacings:
    print(f"\nProcessing spacing: {spacing}")

    # Recreate the source space for this spacing
    src = mne.setup_source_space(
        'fsaverage', spacing=spacing, subjects_dir=subjects_dir, add_dist=False, verbose=False
    )

    for sub_id in sub_ids:
        print(f"  Calculating resolution metrics for {sub_id}...")

        # Define paths to previously computed forward and inverse models
        fwd_fname = os.path.join(fwd_dir, f"{sub_id}_{spacing}-fwd.fif")
        inv_fname = os.path.join(inv_dir, f"{sub_id}_{spacing}-inv.fif")

        # Load models
        fwd = mne.read_forward_solution(fwd_fname, verbose=False)
        inv_op = mne.minimum_norm.read_inverse_operator(inv_fname, verbose=False)

        # Ensure forward solution is restricted to fixed orientations
        # (normal to the cortical surface) for consistency in resolution metric derivation.
        fwd = mne.convert_forward_solution(
            fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=False
        )

        for method in methods:
            # Construct the resolution matrix (W * L)
            resmat = make_inverse_resolution_matrix(
                fwd, inv_op, method=method, lambda2=lambda2, verbose=False
            )

            # Compute Point-Spread Function metrics (PSF: columns of resolution matrix)
            # PLE measures distance between true source and estimated peak
            stc_psf_ple = resolution_metrics(resmat, src, function='psf', metric='peak_err', verbose=False)
            # SD measures the spatial extent/spread of the estimate
            stc_psf_sd = resolution_metrics(resmat, src, function='psf', metric='sd_ext', verbose=False)

            # Compute Cross-Talk Function metrics (CTF: rows of resolution matrix)
            stc_ctf_ple = resolution_metrics(resmat, src, function='ctf', metric='peak_err', verbose=False)
            stc_ctf_sd = resolution_metrics(resmat, src, function='ctf', metric='sd_ext', verbose=False)

            # Save the resulting SourceEstimates
            base_outname = os.path.join(out_dir, f"{sub_id}_{spacing}_{method}")

            stc_psf_ple.save(f"{base_outname}_PSF_PLE", overwrite=True, verbose=False)
            stc_psf_sd.save(f"{base_outname}_PSF_SD", overwrite=True, verbose=False)
            stc_ctf_ple.save(f"{base_outname}_CTF_PLE", overwrite=True, verbose=False)
            stc_ctf_sd.save(f"{base_outname}_CTF_SD", overwrite=True, verbose=False)

            # Free memory (resolution matrices can be quite large)
            del resmat
