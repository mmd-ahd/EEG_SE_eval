"""
Inverse Operator Generation.

This script constructs the minimum-norm inverse operators for each subject.
It combines the individual forward solutions (Leadfields) with the empirical
noise covariance matrices.

The resulting inverse operators are later used to project scalp EEG data
onto the cortical source space. We force a fixed source orientation
(normal to the cortical mantle).
"""

import os
import mne
from mne.minimum_norm import make_inverse_operator, write_inverse_operator

# Define directory paths
dataset_dir = r"ds003505\derivatives\eeglab-v14.1.1"
derivatives_dir = r"ds003505\derivatives"
cov_dir = os.path.join(derivatives_dir, 'covariances')
fwd_dir = os.path.join(derivatives_dir, 'forward_solutions')
out_dir = os.path.join(derivatives_dir, 'inverse_operators')

os.makedirs(out_dir, exist_ok=True)

# List of subject IDs to process
sub_ids = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-06', 'sub-07', 'sub-08',
    'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15',
    'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'
]

# Source space spacings to iterate over
spacings = ['ico4', 'ico5', 'oct6']

# Create inverse operators for all combinations of spacing and subjects
for spacing in spacings:
    print(f"Creating inverse operators for spacing: {spacing}")
    for sub_id in sub_ids:
        print(f"  Processing {sub_id}...")

        # Define paths for required inputs
        cov_fname = os.path.join(cov_dir, f"{sub_id}-cov.fif")
        fwd_fname = os.path.join(fwd_dir, f"{sub_id}_{spacing}-fwd.fif")
        data_path = os.path.join(dataset_dir, sub_id, 'eeg', f'{sub_id}_task-faces_desc-preproc_eeg.set')

        # Load EEGLAB data to extract sensor info
        epochs = mne.read_epochs_eeglab(data_path, verbose=False)
        epochs.set_eeg_reference(projection=True, verbose=False)

        # Load the noise covariance matrix and the forward solution
        noise_cov = mne.read_cov(cov_fname, verbose=False)
        fwd = mne.read_forward_solution(fwd_fname, verbose=False)

        # Convert forward solution to fixed orientation (normal to the cortical surface)
        # Note: Fixed orientation is preferred for these specific resolution metrics
        fwd_fixed = mne.convert_forward_solution(
            fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=False
        )

        # Compute the inverse operator
        # Depth weighting is disabled (depth=None) as it can interact unpredictably with resolution metrics
        inv_op = make_inverse_operator(
            epochs.info, fwd_fixed, noise_cov, loose=0, depth=None, verbose=False
        )

        # Save the inverse operator to disk
        inv_fname = os.path.join(out_dir, f"{sub_id}_{spacing}-inv.fif")
        write_inverse_operator(inv_fname, inv_op, overwrite=True, verbose=False)
