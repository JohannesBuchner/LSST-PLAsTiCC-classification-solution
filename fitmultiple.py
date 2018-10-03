import numpy
import scipy.stats
import sys
import matplotlib.pyplot as plt

time, flux, flux_error = numpy.loadtxt(sys.argv[1]).transpose()
time = time - time[0] 

plt.errorbar(time, flux, yerr=flux_error, ls=' ', marker='x')
plt.ylabel('flux')
plt.xlabel('time')
plt.yscale('log')
plt.savefig('fitsingle_data.pdf', bbox_inches='tight')
plt.close()

c0 = numpy.polynomial.chebyshev.Chebyshev((1,0,0))
c2 = numpy.polynomial.chebyshev.Chebyshev((0,0,1))

def fit_logline(time, flux, flux_error):
	x, y, y_err = time, numpy.log10(flux), numpy.log10(flux_error)
	slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)

	t = numpy.linspace(x[0], x[-1], 400)
	plt.plot(t, 10**(slope * t + intercept), '--', color='k')
	return slope, intercept

lslopes = []
rslopes = []
# cross-validate or bootstrap
for bs in range(20):
	i = get_indices(time)
	time_bs = time[i]
	flux_bs_resampled = numpy.random.normal(flux[i], flux_error[i])
	flux_bs, flux_error_bs = flux[i], flux_error[i]
	
	# find maximum
	peak = numpy.argmax(flux_bs)
	
	# fit left part
	if len(time_bs[0:peak+1]) > 2:
		lslopes.append(fit_logline(time_bs[0:peak+1], flux_bs[0:peak+1], flux_error_bs[0:peak+1]))
	else:
		lslopes.append((numpy.random.normal(0, 5), numpy.nan, numpy.nan, numpy.nan))
	# fit right part
	if len(time_bs[peak:]) > 2:
		rslopes.append(fit_logline(time_bs[peak:], flux_bs[peak:], flux_error_bs[peak:]))
	else:
		rslopes.append((numpy.random.normal(0, 5), numpy.nan, numpy.nan, numpy.nan))

for l, r in zip(lslopes, rslopes):
	print('%.3f\t%.3f\t%.3f\t%.3f' % l, '%.3f\t%.3f\t%.3f\t%.3f' % r)
# concavity: compare central residual to edge residuals (chebyshev weights?)
plt.errorbar(time, flux, yerr=flux_error, ls=' ', marker='x')
plt.ylabel('flux')
plt.xlabel('time')
#plt.yscale('log')
plt.savefig('fitsingle_fit.pdf', bbox_inches='tight')
plt.close()






