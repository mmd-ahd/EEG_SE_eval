"""
ROI-Based Metrics Data Extraction.

This script loads the previously computed whole-brain resolution metrics (PLE and SD 
for both PSF and CTF) and extracts the mean values within specific functional Regions 
of Interest (ROIs). 

The ROIs are defined using the HCP-MMP1 parcellation on the 'fsaverage' template:
- V1 (Primary Visual Cortex)
- Face Network (FFC, PIT)
- Motion Network (MT, MST)

The extracted data is compiled into a tidy pandas DataFrame and exported to a CSV 
file formatted for subsequent statistical analysis.
"""

import os
import mne
import numpy as np
import pandas as pd

derivatives_dir = r"ds003505\derivatives"
metrics_dir = os.path.join(derivatives_dir, 'resolution_metrics')
subjects_dir = os.path.join(derivatives_dir, 'subjects')
out_dir = os.path.join(metrics_dir, 'roi_stats')
os.makedirs(out_dir, exist_ok=True)

sub_ids = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-06', 'sub-07', 'sub-08',
    'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15',
    'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'
]

SPACING = 'ico5'
methods = ['MNE', 'dSPM', 'sLORETA', 'eLORETA']
conditions = ['PSF_PLE', 'PSF_SD', 'CTF_PLE', 'CTF_SD']

roi_config = {
    'V1': ['V1'],
    'Face': ['FFC', 'PIT'],
    'Motion': ['MT', 'MST']
}

mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, verbose=False)
all_labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', subjects_dir=subjects_dir)

target_labels = {}
for net_name, parts in roi_config.items():
    lh_parts = [l for l in all_labels if any(f'_{p}_' in l.name for p in parts) and l.hemi == 'lh']

    if lh_parts:
        merged = lh_parts[0].copy()
        for l in lh_parts[1:]:
            merged += l
        merged.name = net_name
        target_labels[net_name] = merged

data_rows = []

for sub_id in sub_ids:
    formatted_sub = sub_id.replace('sub-', 'S')
    
    for method in methods:
        for condition in conditions:
            stc_fname = os.path.join(metrics_dir, f"{sub_id}_{SPACING}_{method}_{condition}")
            
            try:
                stc = mne.read_source_estimate(stc_fname)
                
                for roi_name, label in target_labels.items():
                    stc_roi = stc.in_label(label)
                    mean_val = np.mean(stc_roi.data)
                    
                    data_rows.append({
                        'Subject': formatted_sub,
                        'Method': method,
                        'ROI': roi_name,
                        'Condition': condition,
                        'Value (cm)': round(mean_val, 4)
                    })
                    
            except Exception as e:
                print(f"Warning: Could not process {formatted_sub} - {method} - {condition}. Error: {e}")

df = pd.DataFrame(data_rows)
csv_out_path = os.path.join(out_dir, f'ROI_Metrics_{SPACING}.csv')
df.to_csv(csv_out_path, index=False)

print(f"Successfully extracted data and saved to: {csv_out_path}")