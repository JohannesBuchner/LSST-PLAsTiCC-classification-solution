
Insights
-----------

* There are large flux calibration issues, leading to negative fluxes
* Bootstrapping of data points (randomized within flux errors) --> use only the ones >0 && detected
* LC inputs: 
  * Bootstrap instances, randomized within flux errors, randomized with redshift
  * Folded with Lomb-Scargle period and peak shifted to 0
* for each filter:
  * find maximum and nearest minima in time. Fit powerlaw slopes (least square fit)
  * quality of the powerlaw descriptions, maximum flux
  * concavity: compare central residual to edge residuals (chebyshev weights?)
* find maximum over all filters
  * compute peak colors
* fit powerlaw plane to color lightcurve (2d least square fit)
* Empirical light curve quantities: variance, skew, kurtosis, fourier components (2/1, 3/1), Shapiro-Wilk test statistic, Lomb-Scargle period, nu(folded LC), cum sum index(folded LC), Stetson K index, IQR, fraction above mean, Fourier decomposition amplitude, 0.1,0.9 slope (folded LC) etc.
  (from https://www.aanda.org/articles/aa/pdf/2016/03/aa27188-15.pdf)

Potentially too slow methods:
* Gaussian processes?
* nfft? pyNFFT
Another issue is that often there are very few data points (<5), so it is difficult to make any statements.

Template fitting? normalize data (peak time, peak value), create large library of functions, evaluate likelihood. Train random forest to group fn-L strength into classes.
This basically approximates what a DNN should do, but might cope better with heteregeneous sampling.


* Learn color and time series evolution simultaneously?
* What about objects appearing near each other (streaks?)


Feature creation
--------------------
FATS, tsfresh libraries

a = FATS.FeatureSpace(Data='all', featureList=None) 
a = a.calculateFeature(lc)


Classes
---------

galactic:   6 16 53 65 92
exgalactic: 15 42 52 62 64 67 88 90 95 

with extract_lcs.py

6: shape extremely close to two powerlaws slope
15: SN -- powerlaw slopes
16: flat with dips -- 3 sigma clipping -- fraction of outliers below 5 sigma
42: SN, but flatter late slope (P?)
52: similar to SN, but short visibility, very flat. Faint. Novae?
53: strongly periodic, almost sinosoidal.
62: SN powerlaws, but has smoother bend
64: few examples. brief. very steep decline
65: very sporadic peaks, lots of scatter. negative flux-time slope. inverted or very flat SED? AGN? no, galactic
67: has negative bend
88: multiple peaks, lots of structure/scatter. Exists for longer
90: SN again, bending
92: similar to 16, quite blue SED
95: bending SN with late break/plateau

SN-like:
6: falls and rises with same speed in all bands
15: diversity in fall/rise slopes across bands
42: rising slope is steep in UV flatter in IR
62: falling slope flatter in general (except UV)
90: falling slope flatter in general (except IR)
95: falling slope flatter in general (except UV)
52: differs from 42 in flux-time slopes

6: rising optical, UV enhanced, IR decrease
15: declining in optical, very strong UV
42: flat to strongly increasing in optical. IR decrease
62: optical increasing, UV excess. IR decrease
67: 
90: opt increasing. UV excess. IR decrease
95: opt increasing. UV excess. IR flat or decrease.

galactic:
6: two powerlaws of same slope, T const, 
16: differs in 15 in slopes, Temp, blueratio of avg SED, #dips, #peaks
53: has different skew than 16
65:
92:



Machine learning
-------------------

* Train random forest on computed features


* Train random forest on clean data -- use only specz
* Rank objects by # of data points, bootstrap dispersion in quantities (mahalanobis distance)
* Choose best half
* Train on median bootstrapped quantities


* For each class, find best, clean instances with lowest dispersion in quantities
* t-SNE or PCA for each class to find diversity
* sample uniformly on whitened structure
* use this training sample for training the random forest


* Outlier detection!


Outdated insights
--------------------

* [X] There are groups of objects with the same cadence/sampling. We can analyse them together! (550 objects) --> actually this is not that common


remove milky way extinction: 
u_factor = 0.0092394833
g_factor = 0.0336530868
r_factor = 0.0938432007
i_factor = 0.1868623254
z_factor = 0.2921024403
y_factor = 0.4306847762




Prior distribution for each class
----------------------------------
* mix in depending on # of valid data points



