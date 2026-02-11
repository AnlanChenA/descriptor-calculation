# descriptor-calculation
1. identify the spatial locations of each agglomerate while allowing for the presence of dispersed particles, and
2. Calculate a descriptor, Ifiller, that describe the system

Naming: VF_2_ParVF_1_Agg_2 indicates a particle volume fraction of 2%, a dispersed (non-clustered) particle volume fraction of 1%, and two agglomerations.

# Core Functionality --- cluster_w_noise.py

The script aims to identify the spatial locations of each agglomerate while allowing for the presence of dispersed particles, consistent with the system information encoded in the naming convention.

This script performs automated clustering of particle distributions using DBSCAN with periodic boundary conditions. It reads binary particle masks from .mat files, converts them into particle coordinates, and applies a grid search over DBSCAN parameters to identify clustering results that match a prescribed number of clusters while satisfying a volume-fraction–based consistency criterion. For each sample, clustering labels are saved to disk, a global summary table of parameters used is saved, and selected results are visualized together after all samples have been processed. The saved ‘cluster_label.txt’ is further used to calculate descriptors of the system.

These are two examples of clustering result:

