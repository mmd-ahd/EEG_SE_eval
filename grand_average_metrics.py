"""
Grand Average Metrics Computation.

This script aggregates the individual subject resolution metrics (PLE and SD
for both PSF and CTF) to compute group-level "Grand Averages".

The averaged SourceEstimates are saved and subsequently used for visualization
(e.g., projecting onto the fsaverage brain) and ROI-based statistical comparisons.
"""

import os
import mne
import numpy as np

# Define directories
derivatives_dir = r"ds003505\derivatives"
metrics_dir = os.path.join(derivatives_dir, 'resolution_metrics')
subjects_dir = os.path.join(derivatives_dir, 'subjects')

# Directories for outputs and visualizations
ga_dir = os.path.join(metrics_dir, 'grand_averages')
screenshots_dir = os.path.join(metrics_dir, 'screenshots')
os.makedirs(ga_dir, exist_ok=True)
os.makedirs(screenshots_dir, exist_ok=True)

# List of subject IDs
sub_ids = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-06', 'sub-07', 'sub-08',
    'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15',
    'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'
]

# Analysis settings
# We focus the grand average on 'ico5' spacing as it provides a good balance
# of resolution and computational feasibility for visualizations.
SPACING = 'ico5'
methods = ['MNE', 'dSPM', 'sLORETA', 'eLORETA']
metrics = ['PSF_PLE', 'PSF_SD', 'CTF_PLE', 'CTF_SD']

# Dictionary to hold the computed grand averages in memory
ga_stcs = {}

for method in methods:
    for metric in metrics:
        print(f"Computing Grand Average for: {method} - {metric}")
        stc_list = []

        # Load individual source estimates
        for sub_id in sub_ids:
            fname = os.path.join(metrics_dir, f"{sub_id}_{SPACING}_{method}_{metric}")

            try:
                # read_source_estimate automatically adds the '-rh.stc' / '-lh.stc' extensions
                stc = mne.read_source_estimate(fname)
                stc_list.append(stc)
            except Exception as e:
                print(f"  Warning: Could not read data for {sub_id} ({e})")

        if not stc_list:
            print(f"  Skipping {method} {metric} - No data found.")
            continue

        # Create a copy of the first subject's STC structure to hold the average
        ga_stc = stc_list[0].copy()

        # Extract the underlying data arrays (vertices x timepoints)
        # Note: Timepoints here actually represent the 1 discrete value for the metric
        all_data = np.array([s.data for s in stc_list])

        # Compute the mean across subjects (axis 0)
        ga_stc.data = np.mean(all_data, axis=0)

        # Save the resulting Grand Average to disk
        ga_fname = os.path.join(ga_dir, f"GA_{SPACING}_{method}_{metric}")
        ga_stc.save(ga_fname, overwrite=True, verbose=False)
        print(f"  Saved: {ga_fname}")

        # Store in dictionary
        ga_stcs[f"{method}_{metric}"] = ga_stc
