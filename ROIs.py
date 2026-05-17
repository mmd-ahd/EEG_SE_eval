import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

derivatives_dir = r"ds003505\derivatives"
metrics_dir = os.path.join(derivatives_dir, 'resolution_metrics')
subjects_dir = os.path.join(derivatives_dir, 'subjects')

roi_definitions = {
    'V1': ['V1'], 
    'Face': ['PIT', 'FFC'], 
    'Motion': ['MT', 'MST']
}

sub_ids = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-06', 'sub-07', 'sub-08', 
    'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15', 
    'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'
]

SPACING = 'ico5' 
methods = ['MNE', 'dSPM', 'sLORETA', 'eLORETA']
metrics = ['PSF_PLE', 'PSF_SD', 'CTF_PLE', 'CTF_SD']

sns.set_theme(style="white", rc={"axes.grid": True, "grid.linestyle": ":", "grid.color": ".8"})
palette = sns.color_palette("mako", n_colors=len(methods))

src = mne.setup_source_space(
    'fsaverage', spacing=SPACING, subjects_dir=subjects_dir, add_dist=False, verbose=False
)

mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, verbose=False)
all_labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', subjects_dir=subjects_dir)

final_labels = []
label_names = []

for net_name, parts in roi_definitions.items():
    lh_parts = [l for l in all_labels if any(f'_{p}_' in l.name for p in parts) and l.hemi == 'lh']
    
    if lh_parts:
        merged = lh_parts[0].copy()
        for l in lh_parts[1:]:
            merged += l
        merged.name = net_name
        final_labels.append(merged)
        label_names.append(net_name)

results = []

for sub_id in sub_ids:
    for method in methods:
        for metric in metrics:
            fname = os.path.join(metrics_dir, f"{sub_id}_{SPACING}_{method}_{metric}-lh.stc")
            
            try:
                stc = mne.read_source_estimate(fname)
                if stc.data.max() < 1.0:
                    stc.data *= 100.0
                
                roi_values = mne.extract_label_time_course(
                    stc, final_labels, src, mode='mean', verbose=False
                )
                
                for i, val in enumerate(roi_values):
                    results.append({
                        'Subject': sub_id,
                        'Method': method,
                        'Metric': metric,
                        'ROI': label_names[i],
                        'Value': val[0]
                    })
            except Exception:
                pass 

df = pd.DataFrame(results)

fig, axes = plt.subplots(2, 2, figsize=(16, 9))

plot_map = {
    (0,0): 'PSF_PLE', (0,1): 'CTF_PLE',
    (1,0): 'PSF_SD',  (1,1): 'CTF_SD'
}

for (row, col), metric_name in plot_map.items():
    ax = axes[row, col]
    subset = df[df['Metric'] == metric_name]
    
    if subset.empty:
        continue
        
    sns.barplot(
        data=subset, x='ROI', y='Value', hue='Method', 
        ax=ax, palette=palette, alpha=0.85, errorbar='se', 
        capsize=0.08, edgecolor='white', linewidth=1.2
    )
    
    ax.set_title(metric_name.replace('_', ' '), fontsize=16, fontweight='bold', pad=12)
    ax.set_ylabel('Error / Deviation (cm)', fontsize=13, fontweight='500')
    ax.set_xlabel('')
    
    sns.despine(ax=ax, top=True, right=True)
    
    if (row, col) == (0, 0):
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:len(methods)], labels[:len(methods)], title='Method', 
                  loc='upper right', framealpha=0.9, edgecolor='lightgray')
    else:
        if ax.get_legend() is not None:
            ax.legend_.remove()

plt.tight_layout(rect=[0, 0.02, 1, 1])

out_file = os.path.join(metrics_dir, f'ROI_Statistical_Comparison_Modern_{SPACING}.png')
plt.savefig(out_file, dpi=600, facecolor='white')
plt.show()

summary = df.groupby(['ROI', 'Method', 'Metric'])['Value'].agg(['mean', 'sem'])