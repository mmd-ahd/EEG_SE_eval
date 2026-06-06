"""
Whole-Source-Space Statistical Analysis of Resolution Metrics.

This script performs non-parametric cluster-based permutation testing
across the entire cortical surface.

It loops over all defined metrics and computes all pairwise comparisons 
between the 4 methods using 1-sample tests on their differences.
"""

import os
import itertools
import numpy as np
import scipy.stats
import mne
from mne.stats import permutation_cluster_1samp_test

derivatives_dir = r"ds003505\derivatives"
metrics_dir = os.path.join(derivatives_dir, 'resolution_metrics')
subjects_dir = os.path.join(derivatives_dir, 'subjects')
stats_dir = os.path.join(metrics_dir, 'whole_brain_stats')
os.makedirs(stats_dir, exist_ok=True)

sub_ids = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-06', 'sub-07', 'sub-08',
    'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15',
    'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'
]
n_subjects = len(sub_ids)

SPACING = 'ico5'
metrics = ['PSF_PLE', 'PSF_SD', 'CTF_PLE', 'CTF_SD']
methods = ['MNE', 'dSPM', 'sLORETA', 'eLORETA']

src = mne.setup_source_space(
    'fsaverage', spacing=SPACING, subjects_dir=subjects_dir, add_dist=False, verbose=False
)
adjacency = mne.spatial_src_adjacency(src)
n_vertices = adjacency.shape[0]

t_thresh = scipy.stats.t.ppf(1 - 0.05 / 2, df=n_subjects - 1)
method_pairs = list(itertools.combinations(methods, 2))

for metric in metrics:
    data = np.zeros((n_subjects, len(methods), n_vertices))
    template_stc = None

    for i, method in enumerate(methods):
        for j, sub_id in enumerate(sub_ids):
            fname = os.path.join(metrics_dir, f"{sub_id}_{SPACING}_{method}_{metric}")
            stc = mne.read_source_estimate(fname)
            
            if template_stc is None:
                template_stc = stc.copy()
                
            data[j, i, :] = stc.data[:, 0]

    for m1, m2 in method_pairs:
        idx1 = methods.index(m1)
        idx2 = methods.index(m2)
        diff_data = data[:, idx1, :] - data[:, idx2, :]
        sig_data = np.zeros(n_vertices)

        if not np.allclose(diff_data, 0):
            clu_pairwise = permutation_cluster_1samp_test(
                diff_data, 
                adjacency=adjacency, 
                n_permutations=10000, 
                threshold=t_thresh, 
                out_type='indices',
                n_jobs=-1
            )

            t_obs, clusters, cluster_pv, H0 = clu_pairwise

            if len(clusters) > 0:
                good_cluster_inds = np.where(cluster_pv < 0.05)[0]
                for i_clu in good_cluster_inds:
                    clu_idx = clusters[i_clu]
                    sig_data[clu_idx] = t_obs[clu_idx]

        stc_pairwise_sig = template_stc.copy()
        stc_pairwise_sig.data = sig_data[:, np.newaxis]
        out_pairwise = os.path.join(stats_dir, f"Stats_Pairwise_{m1}_vs_{m2}_{metric}")
        stc_pairwise_sig.save(out_pairwise, overwrite=True)

print("All pairwise statistical processing complete!")