"""
Histograms and Statistical Summaries of Resolution Metrics.

This script visualizes the distribution of spatial resolution metrics
(Peak Localization Error and Spatial Deviation) across the entire cortical
surface for the Grand Average data.

It generates a grid of histograms mapped to the four inverse methods and
four computed metrics, annotating each distribution with its Mean, Standard
Deviation, and Median to allow for quantitative comparisons of estimator performance.
"""

import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setup directories
derivatives_dir = r"ds003505\derivatives"
metrics_dir = os.path.join(derivatives_dir, 'resolution_metrics')
ga_dir = os.path.join(metrics_dir, 'grand_averages')
stats_dir = os.path.join(metrics_dir, 'statistics')

os.makedirs(stats_dir, exist_ok=True)

# Configuration
SPACING = 'ico4' # Note: Uses ico4 which generally provides a smoother distribution for histograms
methods = ['MNE', 'dSPM', 'sLORETA', 'eLORETA']
metrics = ['PSF_PLE', 'CTF_PLE', 'PSF_SD', 'CTF_SD']

# Configure seaborn styling
sns.set_theme(style="white", rc={"axes.grid": True, "grid.linestyle": ":", "grid.color": ".8"})
palette = sns.color_palette("mako", n_colors=len(methods))

# Initialize matplotlib grid
print("Generating histograms of cortical distributions...")
fig, axes = plt.subplots(nrows=len(methods), ncols=len(metrics), figsize=(18, 10), sharex='col')

for row, method in enumerate(methods):
    method_color = palette[row]

    for col, metric in enumerate(metrics):
        ax = axes[row, col]
        ga_fname = os.path.join(ga_dir, f"GA_{SPACING}_{method}_{metric}-lh.stc")

        try:
            # Load Grand Average STC and extract all cortical vertices as a flat array
            ga_stc = mne.read_source_estimate(ga_fname)
            data = ga_stc.data.flatten()

            # Convert default units to cm for readability if necessary
            if data.max() < 1.0:
                 data *= 100.0

            # Calculate summary statistics
            mean_val = np.mean(data)
            std_val = np.std(data)
            median_val = np.median(data)

            # Configure histogram bins and labels
            bins = np.linspace(0, 10.0, 40)
            xlabel = 'Localization Error (cm)' if 'PLE' in metric else 'Spatial Deviation (cm)'

            # Disable Kernel Density Estimate (KDE) if variance is near 0 to avoid calculation errors
            use_kde = True if std_val > 0.01 else False

            # Plot the histogram
            sns.histplot(
                data, bins=bins, kde=use_kde, color=method_color,
                edgecolor='white', linewidth=0.5, alpha=0.7, ax=ax,
                line_kws={'linewidth': 2} if use_kde else None
            )

            # Overlay a dashed line representing the Mean
            ax.axvline(mean_val, color='#003049', linestyle='--', linewidth=2, alpha=0.9)

            # Add text box containing the summary statistics
            stat_text = f"$\\mu$: {mean_val:.1f} cm\n$\\sigma$: {std_val:.1f} cm\n$M$: {median_val:.1f} cm"
            ax.text(0.95, 0.85, stat_text, transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='white', edgecolor='lightgray', alpha=0.85))

            # Clean up axes (remove y-axis numbers since this is comparative)
            ax.set_ylabel('')
            ax.set_yticks([])

            # Set grid labels (Metrics on top, Methods on left)
            if row == 0:
                ax.set_title(metric.replace('_', ' '), fontsize=16, fontweight='bold', pad=15)
            if col == 0:
                ax.set_ylabel(method, fontsize=16, fontweight='bold', labelpad=15)
            if row == len(methods) - 1:
                ax.set_xlabel(xlabel, fontsize=14, fontweight='500', labelpad=10)
            else:
                ax.set_xlabel('')

            # Remove bounding box
            sns.despine(ax=ax, left=True, top=True, right=True)

        except Exception as e:
            # Handle missing data gracefully
            ax.text(0.5, 0.5, "Data Missing", ha='center', va='center', color='gray')
            ax.axis('off')
            print(f"Warning: Could not process {method}_{metric}: {e}")

# Apply final layout adjustments
plt.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.08, wspace=0.1, hspace=0.2)

# Save the figure
out_fig = os.path.join(stats_dir, f'Histograms_Modern_{SPACING}.png')
plt.savefig(out_fig, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved histogram plot to: {out_fig}")
