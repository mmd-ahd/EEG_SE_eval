import os
import mne
import numpy as np

derivatives_dir = r"ds003505\derivatives"
metrics_dir = os.path.join(derivatives_dir, 'resolution_metrics')
subjects_dir = os.path.join(derivatives_dir, 'subjects')

ga_dir = os.path.join(metrics_dir, 'grand_averages')
screenshots_dir = os.path.join(metrics_dir, 'screenshots')
os.makedirs(ga_dir, exist_ok=True)
os.makedirs(screenshots_dir, exist_ok=True)

sub_ids = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-06', 'sub-07', 'sub-08', 
    'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15', 
    'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'
]

SPACING = 'ico5'
methods = ['MNE', 'dSPM', 'sLORETA', 'eLORETA']
metrics = ['PSF_PLE', 'PSF_SD', 'CTF_PLE', 'CTF_SD']

ga_stcs = {} 

for method in methods:
    for metric in metrics:
        stc_list = []
        for sub_id in sub_ids:
            fname = os.path.join(metrics_dir, f"{sub_id}_{SPACING}_{method}_{metric}")
            
            try:
                stc = mne.read_source_estimate(fname)
                stc_list.append(stc)
            except Exception:
                pass
        
        if not stc_list:
            continue
            
        ga_stc = stc_list[0].copy()
        
        all_data = np.array([s.data for s in stc_list])
        ga_stc.data = np.mean(all_data, axis=0)
        
        ga_fname = os.path.join(ga_dir, f"GA_{SPACING}_{method}_{metric}")
        ga_stc.save(ga_fname, overwrite=True, verbose=False)
        
        ga_stcs[f"{method}_{metric}"] = ga_stc