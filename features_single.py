import numpy
import scipy.stats
import sys
import matplotlib.pyplot as plt
from astropy import LombScargle

time, flux, flux_error = numpy.loadtxt(sys.argv[1]).transpose()
time = time - time[0]
mag = numpy.log10(flux) * 0.4
lc = numpy.array([mag, time])

plt.errorbar(time, flux, yerr=flux_error, ls=' ', marker='x')
plt.ylabel('flux')
plt.xlabel('time')
plt.yscale('log')
plt.savefig('features_single_data.pdf', bbox_inches='tight')
plt.close()

#* Empirical light curve quantities: variance, skew, kurtosis, fourier components (2/1, 3/1), Shapiro-Wilk test statistic, Lomb-Scargle period, nu(folded LC), cum sum index(folded LC), Stetson K index, IQR, fraction above mean, Fourier decomposition amplitude, 0.1,0.9 slope (folded LC) etc.
variance = mag.std()
skew = scipy.stats.skew(mag)
kurtosis = scipy.stats.kurtosis(mag)
iqr = scipy.stats.iqr(mag)
shapiro_wilk, _p_value = scipy.stats.shapiro(mag)
fracabove = (mag > mag.mean()).mean()

frequency, power = LombScargle(time, flux, flux_error)
best_frequency = frequency[np.argmax(power)]
LS_period = 1.0 / frequency
power0 = power[0]
a, b, c = numpy.argsort(power)[::-1][:3]
power1 = power[a]
power2 = power[b]
power3 = power[c]
R21 = power2 / power1
R31 = power3 / power1
R01 = power.sum() / power1

features = [
	variance, skew,
	kurtosis, iqr, shapiro_wilk, fracabove, 
	LS_period, R21, R31, R01
]




