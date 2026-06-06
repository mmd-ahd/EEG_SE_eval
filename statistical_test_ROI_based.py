"""
ROI-Based Statistical Analysis (Wilcoxon + FDR).

This script loads the extracted ROI metrics and performs non-parametric 
pairwise comparisons (Wilcoxon signed-rank tests) between the four inverse 
methods (MNE, dSPM, sLORETA, eLORETA). 

Because we are doing 6 pairwise comparisons per ROI and per condition, 
we apply the Benjamini-Hochberg False Discovery Rate (FDR) correction 
to the p-values to control for multiple comparisons.
"""

import os
import itertools
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

derivatives_dir = r"ds003505\derivatives"
metrics_dir = os.path.join(derivatives_dir, 'resolution_metrics')
stats_dir = os.path.join(metrics_dir, 'roi_stats')
csv_path = os.path.join(stats_dir, 'ROI_Metrics_ico5.csv')

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print("Error: CSV file not found. Please run the extraction script first.")
    exit()

methods = ['MNE', 'dSPM', 'sLORETA', 'eLORETA']
conditions = ['PSF_PLE', 'PSF_SD', 'CTF_PLE', 'CTF_SD']
rois = ['V1', 'Face', 'Motion']

method_pairs = list(itertools.combinations(methods, 2))
stats_results = []

for condition in conditions:
    for roi in rois:
        df_sub = df[(df['Condition'] == condition) & (df['ROI'] == roi)]
        raw_p_values = []
        valid_pairs = []
        
        for m1, m2 in method_pairs:
            data_m1 = df_sub[df_sub['Method'] == m1].sort_values('Subject')['Value (cm)'].values
            data_m2 = df_sub[df_sub['Method'] == m2].sort_values('Subject')['Value (cm)'].values
            
            if len(data_m1) == 0 or len(data_m2) == 0:
                continue
                
            if np.allclose(data_m1, data_m2):
                p_val = 1.0
            else:
                stat, p_val = wilcoxon(data_m1, data_m2)
            
            raw_p_values.append(p_val)
            valid_pairs.append(f"{m1} vs {m2}")
        
        if len(raw_p_values) > 0:
            reject, fdr_p_values, _, _ = multipletests(raw_p_values, alpha=0.05, method='fdr_bh')
            
            for idx, pair in enumerate(valid_pairs):
                stats_results.append({
                    'Condition': condition,
                    'ROI': roi,
                    'Comparison': pair,
                    'p_raw': raw_p_values[idx],
                    'p_fdr': fdr_p_values[idx],
                    'Significant': reject[idx]
                })

df_stats = pd.DataFrame(stats_results)
stats_out_path = os.path.join(stats_dir, 'ROI_Wilcoxon_FDR_Results.csv')
df_stats.to_csv(stats_out_path, index=False)

print(f"Statistical testing complete. Summary saved to: {stats_out_path}")