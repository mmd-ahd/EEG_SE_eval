"""
Composite Visualization of Pairwise Statistical Clusters.

This script generates a 2x2 composite figure for EVERY method pair 
(e.g., MNE vs eLORETA) displaying the statistically significant clusters.

Updates in this version:
1. 'pos_lims' is used to ensure low-t-statistic clusters do not fade into 
   the white background, forcing them to render with high-contrast colors.
2. Figure dimensions and grid spacing (hspace) have been expanded to entirely 
   prevent colorbars from overlapping with subplot titles.
3. Significance thresholds are marked directly on the colorbars.
"""

import os
import itertools
import mne
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib as mpl

derivatives_dir = r"ds003505\derivatives"
metrics_dir = os.path.join(derivatives_dir, 'resolution_metrics')
subjects_dir = os.path.join(derivatives_dir, 'subjects')
stats_dir = os.path.join(metrics_dir, 'whole_brain_stats')
screenshots_dir = os.path.join(metrics_dir, 'screenshots')
os.makedirs(screenshots_dir, exist_ok=True)

SPACING = 'ico5'
metrics = ['PSF_PLE', 'CTF_PLE', 'PSF_SD', 'CTF_SD']
methods = ['MNE', 'dSPM', 'sLORETA', 'eLORETA']
method_pairs = list(itertools.combinations(methods, 2))

n_subjects = 19
t_thresh = scipy.stats.t.ppf(1 - 0.05 / 2, df=n_subjects - 1)
COLORMAP = 'RdBu_r'

def crop_image(img_array):
    is_data = np.any(img_array < 250, axis=2)
    coords = np.argwhere(is_data)
    if coords.size == 0:
        return img_array
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return img_array[y_min:y_max+1, x_min:x_max+1, :]

try:
    mne.viz.set_3d_backend('pyvistaqt')
except Exception:
    pass

for m1, m2 in method_pairs:
    brain_images = {}
    vlims = {} 

    for metric in metrics:
        fname = os.path.join(stats_dir, f"Stats_Pairwise_{m1}_vs_{m2}_{metric}-lh.stc")
        
        try:
            stc = mne.read_source_estimate(fname)
            max_abs_val = np.abs(stc.data).max()
            
            if max_abs_val < t_thresh:
                clim = dict(kind='value', pos_lims=[0, 1, 2])
                vlims[metric] = None
            else:
                clim = dict(kind='value', pos_lims=[0, t_thresh, max_abs_val])
                vlims[metric] = max_abs_val

            brain = stc.plot(
                subject='fsaverage',
                subjects_dir=subjects_dir,
                hemi='lh',
                views='lateral',
                surface='inflated',
                colormap=COLORMAP,
                clim=clim,
                time_viewer=False,
                show_traces=False,
                colorbar=False, 
                background='white',
                size=(600, 500)
            )

            img = brain.screenshot()
            brain_images[metric] = crop_image(img)
            brain.close() 

        except Exception as e:
            print(f"Warning: Could not render {metric} for {m1} vs {m2}: {e}")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 11),
                             gridspec_kw={'wspace': 0.1, 'hspace': 0.45})

    plot_map = {
        (0,0): 'PSF_PLE', (0,1): 'CTF_PLE',
        (1,0): 'PSF_SD',  (1,1): 'CTF_SD'
    }

    for (row, col), metric in plot_map.items():
        ax = axes[row, col]
        img = brain_images.get(metric, None)
        
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Data Missing", ha='center', va='center', color='gray')
            
        ax.axis('off')
        title = metric.replace('_', ' ')
        ax.set_title(title, fontsize=20, fontweight='bold', pad=15)

        if metric in vlims and vlims[metric] is not None:
            vmax = vlims[metric]
            
            cbar_ax = ax.inset_axes([0.15, -0.12, 0.7, 0.04]) 
            norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
            cmap = plt.get_cmap(COLORMAP)
            mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            
            cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('t-Statistic', fontsize=14, fontweight='bold')
            cbar.ax.tick_params(labelsize=11)
            
            cbar.ax.axvline(t_thresh, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
            cbar.ax.axvline(-t_thresh, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
            
            cbar.ax.text(-0.02, 0.5, f"{m2} Higher", va='center', ha='right', 
                         transform=cbar_ax.transAxes, fontsize=11, fontweight='bold', color='#053061')
            cbar.ax.text(1.02, 0.5, f"{m1} Higher", va='center', ha='left', 
                         transform=cbar_ax.transAxes, fontsize=11, fontweight='bold', color='#67001f')
            
        elif metric in vlims and vlims[metric] is None:
            ax.text(0.5, -0.1, "No Significant Clusters", ha='center', va='center', 
                    transform=ax.transAxes, fontsize=14, fontweight='bold', color='gray')

    fig.text(0.29, 0.95, 'Point-Spread Function (PSF)', ha='center', fontsize=22, fontweight='bold')
    fig.text(0.73, 0.95, 'Cross-Talk Function (CTF)', ha='center', fontsize=22, fontweight='bold')
    fig.text(0.5, 1.02, f'{m1} vs {m2}', ha='center', fontsize=28, fontweight='bold')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.05)
    out_fname = os.path.join(screenshots_dir, f'Whole_Brain_Clusters_{m1}_vs_{m2}_{SPACING}.png')
    plt.savefig(out_fname, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out_fname}")
    plt.close(fig)