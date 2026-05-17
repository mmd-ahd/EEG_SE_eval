import os
import mne

dataset_dir = r"ds003505\derivatives\eeglab-v14.1.1"
derivatives_dir = r"ds003505\derivatives"
out_dir = os.path.join(derivatives_dir, 'covariances')
os.makedirs(out_dir, exist_ok=True)

sub_ids = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-06', 'sub-07', 'sub-08', 
    'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15', 
    'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'
]

baseline = (-0.3, 0)
cov_tmin, cov_tmax = -0.3, 0

for sub_id in sub_ids:
    epochs_list = []

    face_path = os.path.join(dataset_dir, sub_id, 'eeg', f'{sub_id}_task-faces_desc-preproc_eeg.set')
    if os.path.exists(face_path):
        ep_faces = mne.read_epochs_eeglab(face_path, verbose=False)
        ep_faces.set_eeg_reference(projection=True, verbose=False)
        ep_faces.apply_proj()
        ep_faces.apply_baseline(baseline, verbose=False)
        epochs_list.append(ep_faces)

    motion_path = os.path.join(dataset_dir, sub_id, 'eeg', f'{sub_id}_task-motion_desc-preproc_eeg.set')
    if os.path.exists(motion_path):
        ep_motion = mne.read_epochs_eeglab(motion_path, verbose=False)
        ep_motion.set_eeg_reference(projection=True, verbose=False)
        ep_motion.apply_proj()
        ep_motion.apply_baseline(baseline, verbose=False)
        epochs_list.append(ep_motion)

    all_epochs = mne.concatenate_epochs(epochs_list, verbose=False)
    
    noise_cov = mne.compute_covariance(
        all_epochs, tmin=cov_tmin, tmax=cov_tmax, 
        method='shrunk', n_jobs=-1, verbose=False
    )
    
    cov_fname = os.path.join(out_dir, f"{sub_id}-cov.fif")
    mne.write_cov(cov_fname, noise_cov, overwrite=True, verbose=False)