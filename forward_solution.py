import os
import mne

dataset_dir = r"ds003505\derivatives\eeglab-v14.1.1"
derivatives_dir = r"ds003505\derivatives"
out_dir = os.path.join(derivatives_dir, 'forward_solutions')
subjects_dir = os.path.join(derivatives_dir, 'subjects')

os.makedirs(out_dir, exist_ok=True)
os.makedirs(subjects_dir, exist_ok=True)

sub_ids = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-06', 'sub-07', 'sub-08', 
    'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15', 
    'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'
]

spacings = ['ico4', 'ico5', 'oct6']

mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=False)
bem_path = os.path.join(subjects_dir, 'fsaverage', 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
bem = mne.read_bem_solution(bem_path, verbose=False)

for spacing in spacings:
    src = mne.setup_source_space(
        'fsaverage', spacing=spacing, subjects_dir=subjects_dir, add_dist=False, verbose=False
    )
    
    for sub_id in sub_ids:
        data_path = os.path.join(dataset_dir, sub_id, 'eeg', f'{sub_id}_task-faces_desc-preproc_eeg.set')
        epochs = mne.read_epochs_eeglab(data_path, verbose=False)
        
        fwd = mne.make_forward_solution(
            epochs.info, trans='fsaverage', src=src, bem=bem, 
            meg=False, eeg=True, mindist=5.0, n_jobs=-1, verbose=False
        )
        
        fwd_fname = os.path.join(out_dir, f"{sub_id}_{spacing}-fwd.fif")
        mne.write_forward_solution(fwd_fname, fwd, overwrite=True, verbose=False)