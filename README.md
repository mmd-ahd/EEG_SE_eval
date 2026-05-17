# Performance Comparison of L2-Minimum-Norm Estimators for EEG Source Localization in Visual Evoked Potentials

This repository contains the Python scripts and analysis pipeline for the manuscript: **"Performance Comparison of L2-Minimum-Norm Estimators for EEG Source Localization in Visual Evoked Potentials"**. 

The code evaluates and compares the spatial resolution properties of four widely used L2-minimum-norm inverse solutions:
* **MNE** (Minimum Norm Estimate)
* **dSPM** (dynamic Statistical Parametric Mapping)
* **sLORETA** (standardized Low Resolution Brain Electromagnetic Tomography)
* **eLORETA** (exact Low Resolution Brain Electromagnetic Tomography)

The evaluation is based on theoretical resolution metrics derived from the resolution matrix, specifically focusing on the **Point-Spread Function (PSF)** and **Cross-Talk Function (CTF)**. We quantify **Peak Localization Error (PLE)** and **Spatial Deviation (SD)** across the cortical surface and within specific functional Regions of Interest (ROIs) relevant to visual processing (V1, Face network, Motion network).

## Dataset

The analysis is configured to run on the openly available **ds003505** dataset (formatted using BIDS). It utilizes preprocessed EEGLAB datasets (`.set`) for 19 subjects performing Visual Evoked Potential tasks (Faces vs. Motion).

## Dependencies

The scripts are written in Python 3 and heavily rely on the MNE-Python ecosystem. To run the pipeline, you will need:

* `mne` (MNE-Python)
* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `pyvista` and `pyvistaqt` (required for 3D brain rendering)

## Repository Structure & Pipeline Workflow

The scripts are designed to be executed in a specific sequence, moving from forward modeling to statistical visualization.

### 1. Forward and Inverse Modeling
* **`forward_solution.py`**: Computes the forward solution (Leadfield matrix) for multiple source space spacings (`ico4`, `ico5`, `oct6`) using a 3-layer BEM model on the `fsaverage` brain.
* **`covariance_matrix.py`**: Computes the empirical noise covariance matrix using the pre-stimulus baseline period (-300ms to 0ms) concatenated across both Face and Motion tasks.
* **`inverse_operator.py`**: Combines the forward solution and noise covariance to construct the inverse operators for each subject.

### 2. Resolution Metrics Computation
* **`resolution_metrics.py`**: The core analysis script. Computes the inverse resolution matrix for MNE, dSPM, sLORETA, and eLORETA, and extracts the PSF and CTF metrics (PLE and SD).
* **`grand_average_metrics.py`**: Averages the computed resolution metrics across all subjects to generate Grand Average source estimates.

### 3. Region of Interest (ROI) Analysis
* **`ROI_based_metrics.py`**: Extracts and renders the spatial spread (leakage) of PSF and CTF specifically for targeted visual ROIs:
  * **V1** (Primary Visual Cortex)
  * **Face Area** (FFC, PIT)
  * **Motion Area** (MT, MST)
* **`ROIs.py`**: Extracts numerical values for PLE and SD within the specified ROIs, performs group-level statistical comparisons, and generates bar plots using Seaborn.

### 4. Visualization & Plotting
* **`plot_metrics_table.py`**: Generates a comprehensive, 16:9 composite figure showing 3D brain plots of the Grand Average PLE and SD for all four inverse methods.
* **`plot_hist.py`**: Generates histograms and calculates summary statistics (mean, median, standard deviation) for the distribution of localization errors and spatial deviations across the entire cortex.
* **`plot_grand_average_metrics.py`**: A utility script for interactive 3D visualization of specific Grand Average metric maps.

## Usage

1. Ensure your dataset is located in the directory specified by the `dataset_dir` and `derivatives_dir` variables (default: `ds003505\derivatives`).
2. Run the preprocessing and modeling scripts sequentially:
   ```bash
   python forward_solution.py
   python covariance_matrix.py
   python inverse_operator.py
