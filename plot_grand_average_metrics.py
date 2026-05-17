import os
import mne

METHOD = 'MNE' 
METRIC = 'PSF_SD'
SPACING = 'ico5' 

derivatives_dir = r"ds003505\derivatives"
metrics_dir = os.path.join(derivatives_dir, 'resolution_metrics')
subjects_dir = os.path.join(derivatives_dir, 'subjects')
ga_dir = os.path.join(metrics_dir, 'grand_averages')

fname = os.path.join(ga_dir, f"GA_{SPACING}_{METHOD}_{METRIC}-lh.stc")
stc = mne.read_source_estimate(fname)

clim = dict(kind='value', lims=[0, 2.5, 8.0]) if 'PLE' in METRIC else dict(kind='value', lims=[0, 4.0, 8.0])

brain = stc.plot(
    subject='fsaverage', 
    subjects_dir=subjects_dir,
    hemi='lh', 
    views='lateral',    
    surface='inflated', 
    colormap='hot', 
    clim=clim,
    time_viewer=False, 
    show_traces=False,
    colorbar=True,
    size=(800, 600),
    title=f"{METHOD} {METRIC.replace('_', ' ')}"
)

input("Press Enter to close the plot...")