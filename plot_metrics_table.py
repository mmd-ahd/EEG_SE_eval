"""
Composite Visualization Table of Resolution Metrics.

This script generates a comprehensive, publication-ready composite figure
(16:9 aspect ratio) displaying 3D cortical plots of the Grand Average Peak
Localization Error (PLE) and Spatial Deviation (SD) for all four evaluated
inverse methods (MNE, dSPM, sLORETA, eLORETA).

The metrics are rendered on the inflated 'fsaverage' brain template using
the left hemisphere (lateral view) as a representative standard.
"""

import os
import mne
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Setup directory paths
derivatives_dir = r"ds003505\derivatives"
metrics_dir = os.path.join(derivatives_dir, 'resolution_metrics')
subjects_dir = os.path.join(derivatives_dir, 'subjects')

ga_dir = os.path.join(metrics_dir, 'grand_averages')
screenshots_dir = os.path.join(metrics_dir, 'screenshots')
os.makedirs(screenshots_dir, exist_ok=True)

# Configuration for plotting
SPACING = 'ico5'
methods = ['MNE', 'dSPM', 'sLORETA', 'eLORETA']
metrics = ['PSF_PLE', 'CTF_PLE', 'PSF_SD', 'CTF_SD']

# Colormap settings representing spatial extent in centimeters
# Range is set from 0cm to 8cm for optimal contrast across methods
VMIN = 0
VMAX = 8
CLIM = dict(kind='value', lims=[0, 4, 8])
COLORMAP = 'hot'

def crop_image(img_array):
    """
    Crops whitespace from a brain screenshot array to ensure tight packing
    in the final matplotlib grid.
    """
    is_data = np.any(img_array < 250, axis=2)
    coords = np.argwhere(is_data)
    if coords.size == 0:
        return img_array
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return img_array[y_min:y_max+1, x_min:x_max+1, :]

# Explicitly request the pyvistaqt backend for better rendering if available
try:
    mne.viz.set_3d_backend('pyvistaqt')
except Exception:
    pass

brain_images = {}
print("Rendering 3D cortical maps...")

for method in methods:
    for metric in metrics:
        ga_fname = os.path.join(ga_dir, f"GA_{SPACING}_{method}_{metric}-lh.stc")

        try:
            # Read the computed Grand Average source estimate
            ga_stc = mne.read_source_estimate(ga_fname)

            # Plot the metric on the fsaverage surface
            brain = ga_stc.plot(
                subject='fsaverage',
                subjects_dir=subjects_dir,
                hemi='lh',
                views='lateral',
                surface='inflated',
                colormap=COLORMAP,
                clim=CLIM,
                time_viewer=False,
                show_traces=False,
                colorbar=False,
                background='white',
                size=(600, 500)
            )

            # Extract image data, crop whitespace, and store
            img = brain.screenshot()
            brain_images[(method, metric)] = crop_image(img)
            brain.close() # Free rendering resources

        except Exception as e:
            print(f"  Warning: Could not render {method}_{metric}: {e}")

# Construct the composite matplotlib grid
print("Assembling final composite figure...")
fig, axes = plt.subplots(nrows=len(methods), ncols=len(metrics), figsize=(16, 9),
                         gridspec_kw={'wspace': 0.01, 'hspace': 0.01, 'right': 0.92})

# Populate the grid
for i, method in enumerate(methods):
    for j, metric in enumerate(metrics):
        ax = axes[i, j]

        img = brain_images.get((method, metric), None)
        if img is not None:
            ax.imshow(img)

        ax.axis('off')

        # Add column headers (PSF / CTF)
        if i == 0:
            ax.set_title(metric.split('_')[0], fontsize=18, fontweight='bold', pad=5)

        # Add row headers (Inverse Method)
        if j == 0:
            ax.text(-0.05, 0.5, method, va='center', ha='right',
                    transform=ax.transAxes, fontsize=18, fontweight='bold', rotation=90)

# Add super-headers grouping PLE and SD columns
fig.text(0.32, 0.96, 'Peak Localization Error', ha='center', fontsize=22, fontweight='bold')
fig.text(0.74, 0.96, 'Spatial Deviation', ha='center', fontsize=22, fontweight='bold')

# Construct the global vertical colorbar
cbar_ax = fig.add_axes([0.93, 0.1, 0.015, 0.8])
norm = mpl.colors.Normalize(vmin=VMIN, vmax=VMAX)
cmap = plt.get_cmap(COLORMAP)
mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(mappable, cax=cbar_ax)
cbar.set_label('Error / Deviation (cm)', fontsize=16, fontweight='bold', rotation=270, labelpad=25)
cbar.ax.tick_params(labelsize=14)

plt.subplots_adjust(left=0.05, right=0.91, top=0.92, bottom=0.01)

# Save the final result
out_fname = os.path.join(screenshots_dir, f'Metrics_Table_{SPACING}.png')
plt.savefig(out_fname, dpi=600, bbox_inches='tight', facecolor='white')
print(f"Saved visualization to: {out_fname}")
