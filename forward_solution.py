"""
Forward Solution Computation.

This script computes the forward solution (Leadfield matrix) for multiple
source space spacings (ico4, ico5, oct6) using a 3-layer BEM (Boundary Element Method)
model on the 'fsaverage' brain template.

The forward solution relates the source activity in the brain to the EEG signals
recorded at the scalp electrodes.
"""

import os
import mne

# Define directory paths for dataset and output
dataset_dir = r"ds003505\derivatives\eeglab-v14.1.1"
derivatives_dir = r"ds003505\derivatives"
out_dir = os.path.join(derivatives_dir, 'forward_solutions')
subjects_dir = os.path.join(derivatives_dir, 'subjects')

os.makedirs(out_dir, exist_ok=True)
os.makedirs(subjects_dir, exist_ok=True)

# List of subject IDs to process
sub_ids = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-06', 'sub-07', 'sub-08',
    'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15',
    'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'
]

# Source space spacings to iterate over
spacings = ['ico4', 'ico5', 'oct6']

# Download fsaverage template and read the standard 3-layer BEM solution
mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=False)
bem_path = os.path.join(subjects_dir, 'fsaverage', 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
bem = mne.read_bem_solution(bem_path, verbose=False)

# Iterate through each spacing setting to create source spaces
for spacing in spacings:
    print(f"Setting up source space for spacing: {spacing}")
    src = mne.setup_source_space(
        'fsaverage', spacing=spacing, subjects_dir=subjects_dir, add_dist=False, verbose=False
    )

    # Iterate through all subjects and compute their forward solutions
    for sub_id in sub_ids:
        print(f"Computing forward solution for {sub_id} with spacing {spacing}...")

        # Load preprocessed EEGLAB epochs
        data_path = os.path.join(dataset_dir, sub_id, 'eeg', f'{sub_id}_task-faces_desc-preproc_eeg.set')
        epochs = mne.read_epochs_eeglab(data_path, verbose=False)

        # Compute the forward solution
        fwd = mne.make_forward_solution(
            epochs.info, trans='fsaverage', src=src, bem=bem,
            meg=False, eeg=True, mindist=5.0, n_jobs=-1, verbose=False
        )

        # Save the computed forward solution to disk
        fwd_fname = os.path.join(out_dir, f"{sub_id}_{spacing}-fwd.fif")
        mne.write_forward_solution(fwd_fname, fwd, overwrite=True, verbose=False)
        print(f"Saved: {fwd_fname}")
