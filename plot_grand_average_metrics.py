"""
Interactive 3D Visualization of Grand Average Metrics.

This utility script allows for the interactive viewing of specific Grand Average
cortical maps (e.g., MNE PSF Spatial Deviation) on the 'fsaverage' brain template.

It is useful for qualitative inspection of the data before generating the
final composite figures for publication.
"""

import os
import mne

# ------------------------------------------------------------------
# CONFIGURATION
# Modify these variables to change the map being displayed
# ------------------------------------------------------------------
METHOD = 'MNE'      # Options: 'MNE', 'dSPM', 'sLORETA', 'eLORETA'
METRIC = 'PSF_SD'   # Options: 'PSF_PLE', 'CTF_PLE', 'PSF_SD', 'CTF_SD'
SPACING = 'ico5'    # Options: 'ico4', 'ico5', 'oct6'
# ------------------------------------------------------------------

# Setup directories
derivatives_dir = r"ds003505\derivatives"
metrics_dir = os.path.join(derivatives_dir, 'resolution_metrics')
subjects_dir = os.path.join(derivatives_dir, 'subjects')
ga_dir = os.path.join(metrics_dir, 'grand_averages')

# Load the specified Grand Average source estimate
fname = os.path.join(ga_dir, f"GA_{SPACING}_{METHOD}_{METRIC}-lh.stc")
print(f"Loading: {fname}")

try:
    stc = mne.read_source_estimate(fname)
except Exception as e:
    print(f"Error: Could not load the requested file. Make sure it exists.\n{e}")
    exit()

# Configure color limits dynamically based on the type of metric (PLE vs SD)
# PLE values tend to be smaller than Spatial Deviation (SD) values.
clim = dict(kind='value', lims=[0, 2.5, 8.0]) if 'PLE' in METRIC else dict(kind='value', lims=[0, 4.0, 8.0])

# Render the interactive 3D plot
print("Rendering interactive brain window...")
brain = stc.plot(
    subject='fsaverage',
    subjects_dir=subjects_dir,
    hemi='lh',              # Display the left hemisphere
    views='lateral',        # Initial viewing angle
    surface='inflated',     # Inflated surface to see into sulci
    colormap='hot',
    clim=clim,
    time_viewer=False,
    show_traces=False,
    colorbar=True,
    size=(800, 600),
    title=f"{METHOD} {METRIC.replace('_', ' ')}"
)

# Keep the script running so the interactive window doesn't close immediately
input("Press Enter to close the plot and exit the script...")
