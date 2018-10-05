LSST photometric data
===============================

This approach uses custom-engineered feature to aid machine learning.

This data set has some issues:
1) Fluxes can be wrong (very negative) because of wrong subtraction -- the flux errors do not show this
2) Heterogeneous and poor time series sampling. Many object classes show exponential light curve rises/falls. Others show periodic behaviour.
3) Imbalanced object classes.
4) Different photometric bands, making the time series analysis a 2d problem.

To address 2, make_features.py performs bootstrapped least square line fits to get rise and fall slopes and peak fluxes.
This is done for each photometric band. The colors at the peaks are also computed.
Not completely fast: takes 30 minutes for the 7849 training samples on 1 laptop CPU.

make_std_features.py computes some common Lomb-Scargle periodicity, skew etc. lightcurve statistics.
Relatively fast: takes 18 minutes for the 7849 training samples on 1 laptop CPU.

The goal of my approach is to understand the dataset better with relatively blunt tools that still can be understood and interpreted.

Classification
------------------

* random forests because I don't like whitening.
* PCA + K nearest neighbours (K=2, 4, 10, 40, 100)
* combine predictions with small NN or simple weighting

