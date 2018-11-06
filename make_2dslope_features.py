from __future__ import print_function, division
from collections import defaultdict
import os, sys
import numpy
import scipy.stats
import pandas
from numpy import pi, exp, log10
import itertools

def runs_of_ones(bits):
	return [sum(group) for bit, group in itertools.groupby(bits) if bit]

c2 = numpy.polynomial.chebyshev.Chebyshev((0,0,1))
def compute_concavity(x, y, yerr, ymodel):
	resid = y - ymodel
	rel_resid = (y - ymodel) / ymodel
	# normalize to -1 to +1 interval
	xnorm = (x - x[0]) * 2 / (x[-1] - x[0]) - 1
	# variance beyond noise
	excess_variance = (resid**2 - yerr**2).sum() / (ymodel**2).sum()
	excess_variance = ((resid/yerr)**2).mean()
	
	# concavity: weighting to edges
	c2_values = (c2(xnorm) + 1) / 2
	concavity = (c2_values * rel_resid).sum() / c2_values.sum()
	
	return excess_variance, concavity

def LC_slope_singlefit(time, flux, flux_error, z):
	if not numpy.isfinite(z):
		return [numpy.nan]*4
	# get subsampled restframe time
	# at rest frame, more time passed
	time_bs = time / (1 + z)
	flux_bs_resampled = numpy.random.normal(flux, flux_error)
	flux_bs, flux_error_bs = flux, flux_error

	# find maximum
	peak = numpy.argmax(flux_bs)
	peakflux = flux_bs[peak]
	
	# fit left part
	lresult = [numpy.nan, time_bs[peak], 10, 10]
	if len(time_bs[0:peak+1]) > 2:
		lresult = fit_logline(time_bs[0:peak+1], flux_bs[0:peak+1], flux_error_bs[0:peak+1])
	
	# fit right part
	rresult = [numpy.nan, time_bs[peak], 10, 10]
	if len(time_bs[peak:]) > 2:
		rresult = fit_logline(time_bs[peak:], flux_bs[peak:], flux_error_bs[peak:])
	
	# find intercept
	# lresult[0] * t + lresult[1] == rresult[0] * t + rresult[1]
	# (lresult[0] - rresult[0]) * t == rresult[1] - lresult[1]
	# t = (rresult[1] - lresult[1]) / (lresult[0] - rresult[0])
	tintercept = (rresult[1] - lresult[1]) / (lresult[0] - rresult[0])
	peak_intercept = lresult[0] * tintercept + lresult[1]
	useful_fit = numpy.isfinite(peak_intercept) and numpy.isfinite(tintercept)
	#print(prefix, useful_fit, lresult[0], rresult[0], tintercept, time_bs[peak])
	if not useful_fit:
		tintercept = time_bs[peak]
		peak_intercept = flux_bs[peak]
	
	return peakflux, tintercept, lresult[0], rresult[0]


def fit_logline(time, flux, flux_error):
	x, y, y_err = time, log10(flux), log10(flux_error)
	slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
	#excess_variance, concavity = compute_concavity(time, flux, flux_error, ymodel=10**(slope * x + intercept))
	excess_variance, concavity = std_err, r_value

	return [slope, intercept, excess_variance, concavity]

def fit_logplane(x, y, z):
	"""
	Fits z = a*x + b*y + d
	minimizes z distances
	returns a, b
	"""
	A = numpy.matrix([x, y, numpy.ones_like(x)]).T
	b = numpy.matrix(z).T
	#print(A.shape, b.shape)
	#print("A:", A)
	#print("b:", b)
	# solve Ax = b
	#fit = numpy.linalg.solve(A, b)
	try:
		fit = (A.T * A).I * A.T * b
		errors = b - A * fit
		residual = numpy.linalg.norm(errors)

		#print("plane: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
		# so fit * (x, y, 1) = z
		return float(fit[0]), float(fit[1])
	except numpy.linalg.linalg.LinAlgError:
		return numpy.nan, numpy.nan

def fit_logplanetilt(x, y, z):
	"""
	Fits z = a*x + b*y + c*xy + d
	minimizes z distances
	returns a, b, c
	"""
	A = numpy.matrix([x, y, x * y, numpy.ones_like(x)]).T
	b = numpy.matrix(z).T
	#print(A.shape, b.shape)
	#print("A:", A)
	#print("b:", b)
	# solve Ax = b
	#fit = numpy.linalg.solve(A, b)
	try:
		fit = (A.T * A).I * A.T * b
		errors = b - A * fit
		residual = numpy.linalg.norm(errors)

		#print("xy-plane: %f x + %f y + %f xy + %f = z (from %d data points)" % (fit[0], fit[1], fit[2], fit[3], len(z)))
		# so fit * (x, y, 1) = z
		return float(fit[0]), float(fit[1]), float(fit[2])
	except numpy.linalg.linalg.LinAlgError:
		return numpy.nan, numpy.nan, numpy.nan
	
# in units of nm
Tmin = 100
Tmax = 10000
wavenorm = 400 # evaluate the SED at 400nm rest-frame

Twave_grid = numpy.logspace(2, 4, 40).reshape((-1,1))
def bbody_fit(wave, flux):
	if not numpy.isfinite(wave).all():
		return [numpy.nan]*4
	chi2best = 1e300
	best = None
	flux_error = 1

	# also try linear model
	slope, intercept, excess_variance, concavity = fit_logline(log10(wave), flux, flux_error)
	flux_model = 10**(slope * log10(wave) + intercept)
	s = (flux * flux_model / flux_error**2).sum() / (flux_model**2 / flux_error**2).sum()
	chi2_PL = (((flux - s * flux_model) / flux_error)**2).sum()
	#print("after BB fitted PL:", slope, intercept, chi2, chi2best)
	
	wave = wave.reshape((1, -1))
	flux = flux.reshape((1, -1))
	flux_model = wave**-5 / (numpy.exp(1/(wave/Twave_grid)) - 1)
	s = (flux * flux_model / flux_error**2).sum(axis=1) / (flux_model**2 / flux_error**2).sum(axis=1)
	assert len(s) == len(Twave_grid), (s.shape, Twave_grid.shape)
	chi2 = (((flux - s.reshape((-1,1)) * flux_model) / flux_error)**2).sum(axis=1)
	assert len(chi2) == len(Twave_grid), (chi2.shape, Twave_grid.shape)
	i = numpy.argmin(chi2)
	chi2best = float(chi2[i])
	best = [float(Twave_grid[i]), float(s[i])]
	
	return best + [slope, chi2best - chi2_PL]

#def fit_logline(time, flux, flux_error):
#	x, y, y_err = time, log10(flux), log10(flux_error)
#	slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
#	excess_variance, concavity = compute_concavity(time, flux, flux_error, ymodel=10**(slope * x + intercept))
#
#	t = numpy.linspace(x[0], x[-1], 400)
#	return [slope, intercept, excess_variance, concavity]

def get_indices(time):
	# bootstrap
	return numpy.unique(numpy.random.randint(0, len(time), size=len(time)))

def LC_slopes(time, flux, flux_error, specz, photoz, photoz_error):
	breaktimes = []
	slopes = []
	# cross-validate or bootstrap
	for bs in range(20):
		if specz == 0:
			z = 0
		elif numpy.isfinite(specz):
			z = specz
		else:
			z = sample_photoz(photoz, photoz_error)
		
		if len(time) > 0:
		
			i = get_indices(time)
			# get subsampled restframe time
			# at rest frame, more time passed
			time_bs = time[i] / (1 + z)
			flux_bs_resampled = numpy.random.normal(flux[i], flux_error[i])
			flux_bs, flux_error_bs = flux[i], flux_error[i]
		
			# find maximum
			peak = numpy.argmax(flux_bs)
			peakflux = flux_bs[peak]
		else:
			time_bs = [0]
		
			# find maximum
			peak = 0
			peakflux = 1e10
		
		# fit left part
		lresult = [numpy.random.normal(0, 5), time_bs[peak], 10, 10]
		if len(time_bs[0:peak+1]) > 2:
			lresult = fit_logline(time_bs[0:peak+1], flux_bs[0:peak+1], flux_error_bs[0:peak+1])
		
		# fit right part
		rresult = [numpy.random.normal(0, 5), time_bs[peak], 10, 10]
		if len(time_bs[peak:]) > 2:
			rresult = fit_logline(time_bs[peak:], flux_bs[peak:], flux_error_bs[peak:])
		
		# find intercept
		# lresult[0] * t + lresult[1] == rresult[0] * t + rresult[1]
		# (lresult[0] - rresult[0]) * t == rresult[1] - lresult[1]
		# t = (rresult[1] - lresult[1]) / (lresult[0] - rresult[0])
		tintercept = (rresult[1] - lresult[1]) / (lresult[0] - rresult[0])
		peak_intercept = lresult[0] * tintercept + lresult[1]
		
		slopes.append(lresult + rresult + [log10(peakflux), log10(peak_intercept)])
		breaktimes.append(tintercept)

	#for entry in slopes:
	#	for v in entry:
	#		print('%.3f' % v, end='\t')
	#	print()
	
	slopes = numpy.asarray(slopes)
	#goodl = slopes[:,2] > 2
	#goodr = slopes[:,2+4] > 2
	
	# features:
	# (for each of the following: median and iqr)
	# left fits slope
	# left fits variance
	# left fits concavity
	# right fits slope
	# right fits variance
	# right fits concavity
	# peak flux loglinear interpolated
	results = []
	results += []
	for v in slopes[:,0], slopes[:,2], slopes[:,3], slopes[:,0+4], slopes[:,2+4], slopes[:,3+4], slopes[:,9]:
		vmask = numpy.isfinite(v)
		if vmask.any():
			#results += scipy.stats.mstats.mquantiles(v[vmask], [0.25, 0.5, 0.75]).tolist()
			results += [numpy.median(v[vmask]), scipy.stats.iqr(v[vmask])]
		else:
			results += [numpy.nan,numpy.nan]
	return results

## ugrizY EBV effect, assuming a flat spectrum
#extinction_factors = [0.0092394833, 0.0336530868, 0.0938432007, 0.1868623254, 0.2921024403, 0.4306847762]
bands = range(6)

# very roughly read off the LSST passband plot
wavelengths = [357, 475, 621, 755, 879, 1000]
passband_efficiencies = [65, 140, 131, 118, 98, 151]

wavelengths = numpy.array(wavelengths)
passband_efficiencies = numpy.array(passband_efficiencies)


prefix = sys.argv[1]
a = pandas.read_csv(prefix + '.csv')
b = pandas.read_csv(prefix + '_metadata.csv')
a = a.set_index('object_id')
b = b.set_index('object_id')

# slim down table
for col in 'ra', 'decl', 'gal_l', 'gal_b', 'ddf', 'distmod', 'mwebv', 'target':
	try:
		b.pop(col)
	except KeyError:
		pass

e = a.join(b)

# columns:
flux_columns = ['mjd', 'passband', 'flux', 'flux_err', 'detected']
object_columns = ['ra', 'decl', 'gal_l', 'gal_b', 'ddf', 'hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err', 'distmod', 'mwebv', 'target']

mask = b.hostgal_specz > 0
specz = b.hostgal_specz[mask].values
photoz = b.hostgal_photoz[mask].values
del a, b

Nz = 15
zindex = (numpy.log10(photoz+1) * Nz / 0.5).astype(int)
zindex[zindex <  0] = 0
zindex[zindex > Nz-1] = Nz-1
zdists = [[] for i in range(Nz)]
for zi, speczi in zip(zindex, specz):
	zdists[zi].append(speczi)
zdists = [numpy.sort(zdist) for zdist in zdists]
print('zdists:', [len(zdist) for zdist in zdists])

def sample_photoz(photozi, photozi_err, Nsamples=None):
	# original:
	# return numpy.random.normal(photozi, photozi_err, size=Nsamples)
	# draw from empirical distribution
	zindexi = numpy.log10(photozi+1) * Nz / 0.5
	if not zindexi > 0:
		zindexi = 0
	zindexi = min(Nz-1, int(zindexi))
	zdist = zdists[zindexi]
	if len(zdist) < 5:
		return numpy.random.normal(photozi, photozi_err, size=Nsamples)
	else:
		return numpy.random.choice(zdist, replace=True, size=Nsamples)
	

fout = open(prefix + '_colorslope_features.txt', 'w')
header = ""
for color in 'ugrizY':
	for c in "leftslope, leftvar, leftconc, rightslope, rightvar, rightconc, peakfluxi".split(', '):
		header += "%s_%s_median,%s_%s_iqr," % (color, c, color, c)
	header += "%s_peakcolor,%s_peakcolori," % (color, color)

columns = """
Tflux_slope, Tflux_corr,
redrise, redfall,
bluerise, bluefall,
blueconcrise, blueconcfall,
phi_rise, theta_rise, evolrise,
phi_fall, theta_fall, evolfall,
Twave, flux400,
Trise, Tfall, Tratio,
nSEDpeaks, nSEDdips
"""
header += columns.replace("\n", '').replace(" ","")
print("header columns: %d" % header.count(','))
print(header)
fout.write(header + "\n")
linefmt = ('%f,' * (header.count(',')+1)).rstrip(',') + "\n"

for object_id, object_data in e.groupby(e.index.get_level_values(0)):
	print(object_id)
	specz = object_data['hostgal_specz'].values[0]
	photoz = object_data['hostgal_photoz'].values[0]
	photoz_error = object_data['hostgal_photoz_err'].values[0]
	
	if specz == 0:
		z = 0
	elif numpy.isfinite(specz):
		z = specz
	else:
		z = photoz
	
	all_time = object_data['mjd'].values
	all_flux = object_data['flux'].values
	all_flux_error = object_data['flux_err'].values
	all_passband = object_data['passband'].values
	is_detected = object_data['detected'] == 1
	lc_slope_features_all = []
	
	peakfluxes = []
	peaktimes = []

	for passband in bands:
		mask = numpy.logical_and(all_passband == passband, 
			numpy.logical_and(all_flux > 0, is_detected))
		
		flux = all_flux[mask]
		flux_error = all_flux_error[mask]
		time = all_time[mask]

		# create features
		lc_slope_features = LC_slopes(time, flux, flux_error, specz, photoz, photoz_error)
		lc_slope_features_all.append(lc_slope_features)
		
		if len(flux) > 1:
			peakflux, tpeak, lslope, rslope = LC_slope_singlefit(time, flux, flux_error, z)
			peaktimes.append(tpeak)
			peakfluxes.append([wavelengths[passband] / (1 + z), peakflux / passband_efficiencies[passband]])
	
	lc_slope_features_all = numpy.asarray(lc_slope_features_all)
	peak_colors = lc_slope_features_all[:,-2] - lc_slope_features_all[:,-2].max()
	peak_colors_intercept = lc_slope_features_all[:,-1] - lc_slope_features_all[:,-1].max()
	lc_slope_features_all = numpy.hstack((lc_slope_features_all.flatten(), peak_colors, peak_colors_intercept))
	features = lc_slope_features_all.tolist()

	if peaktimes:
		tpeak_lo, tpeak_hi, tpeak = min(peaktimes), max(peaktimes), numpy.median(peaktimes)
	else:
		tpeak_lo, tpeak_hi, tpeak = None, None, all_time[all_flux.argmax()] / (1 + z)
	
	lastt = -100
	deltat = 0.1
	blueratioseries = []
	blueconcseries = []
	redratioseries = []
	dipseries = []
	peakseries = []
	Tseries = []
	Trise = []
	Tfall = []
	Tflux = []
	for t in numpy.unique(all_time[numpy.logical_and(all_flux > 0, is_detected)]):
		if t < lastt + deltat:
			continue
		lastt = t
		mask = numpy.logical_and(numpy.logical_and(all_time >= t, all_time < t + deltat), 
			numpy.logical_and(all_flux > 0, is_detected))
		flux = all_flux[mask]
		passbands = all_passband[mask]
		flux_error = all_flux_error[mask]
		if (passbands == 0).any() and (passbands == 1).any():
			blueratioseries.append([t, float(flux[passbands == 0].mean() / flux[passbands == 1].mean())])
			# measure if the slope is concave or convex 
			a = log10(flux[passbands == 0].mean() / passband_efficiencies[0])
			b = log10(flux[passbands == 1].mean() / passband_efficiencies[1])
			c = log10(flux[passbands == 2].mean() / passband_efficiencies[2])
			la = log10(wavelengths[0])
			lb = log10(wavelengths[1])
			lc = log10(wavelengths[2])
			bexpected = (lb - lc) / (la - lc) * (a - c) + c
			blueconcseries.append([t, float(b - bexpected)])
		if (passbands == 4).any() and (passbands == 5).any():
			redratioseries.append([t, float(flux[passbands == 4].mean() / flux[passbands == 5].mean())])
		
		if mask.sum() > 3:
			trest = t / (1 + z)
			Twave, s, slope, chi2diff = bbody_fit(wavelengths[passbands] / (1 + z), flux / flux.max() / passband_efficiencies[passbands])

			flux400 = s * Twave**-5 / (numpy.exp(1/(wavenorm/Twave)) - 1)
			Tflux.append([float(Twave), float(flux400)])
			
			Tseries.append([trest, Twave, slope, chi2diff])
			if tpeak_lo is not None and trest < tpeak_lo:
				Trise.append(Twave)
			if tpeak_hi is not None and trest > tpeak_hi:
				Tfall.append(Twave)

		# compute runs for number of peaks in the SED
		order = numpy.argsort(passbands)
		flux = flux[order] / passband_efficiencies[passbands][order]
		flux_error = flux_error[order] / passband_efficiencies[passbands][order]
		mask_left_down  = flux[1:-1] < flux[0:-2]
		mask_right_up   = flux[1:-1] < flux[2:]
		ndips = numpy.logical_and(mask_left_down, mask_right_up).sum()
		dipseries.append(ndips)
		mask_left_up    = flux[1:-1] > flux[0:-2]
		mask_right_down = flux[1:-1] > flux[2:]
		npeaks = numpy.logical_and(mask_left_up, mask_right_down).sum()
		peakseries.append(npeaks)

	if len(Tflux) > 3:
		Tflux = numpy.array(Tflux)
		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(log10(Tflux[:,0] / 4000), log10(Tflux[:,1]))
		features += [slope, r_value]
	else:
		features += [numpy.nan, numpy.nan]
	
	if len(redratioseries) > 1:
		tred, redratio = numpy.transpose(redratioseries)
		redrise, redfall = numpy.mean(redratio[tred < tpeak]), numpy.mean(redratio[tred > tpeak])
		features += [redrise, redfall]
	else:
		features += [numpy.nan, numpy.nan]
	
	if len(blueratioseries) > 1:
		tblue, blueratio = numpy.transpose(blueratioseries)
		bluerise, bluefall = numpy.mean(blueratio[tblue < tpeak]), numpy.mean(blueratio[tblue > tpeak])
		features += [bluerise, bluefall]
		blueconcrise, blueconcfall = numpy.mean(numpy.array(blueconcseries)[tblue < tpeak]), numpy.mean(numpy.array(blueconcseries)[tblue > tpeak])
		features += [blueconcrise, blueconcfall]
	else:
		features += [numpy.nan, numpy.nan]
		features += [numpy.nan, numpy.nan]
	
	# find eigenvectors of log(t_rest), log(wave), log(flux) surface
	mask = numpy.logical_and(numpy.logical_and(all_flux > 0, is_detected), all_time / (1 + z) < tpeak)
	if mask.sum() > 6:
		passbands = all_passband[mask]
		phi_rise, theta_rise, evolrise = fit_logplanetilt(all_time[mask] - all_time[mask].max() / (1 + z), 
			log10(wavelengths[passbands] / (1 + z) / 300),
			log10(all_flux[mask] / passband_efficiencies[passbands]))
		phi_rise, theta_rise = fit_logplane(all_time[mask] - all_time[mask].min() / (1 + z), 
			log10(wavelengths[passbands] / (1 + z) / 300),
			log10(all_flux[mask] / passband_efficiencies[passbands]))
	else:
		phi_rise, theta_rise, evolrise = numpy.nan, numpy.nan, numpy.nan
	
	features += [phi_rise, theta_rise, evolrise]

	mask = numpy.logical_and(numpy.logical_and(all_flux > 0, is_detected), all_time / (1+z) > tpeak)
	if mask.sum() > 6:
		passbands = all_passband[mask]
		phi_fall, theta_fall, evolfall = fit_logplanetilt(all_time[mask] - all_time[mask].min() / (1 + z), 
			log10(wavelengths[passbands] / (1 + z) / 300),
			log10(all_flux[mask] / passband_efficiencies[passbands]))
		phi_fall, theta_fall = fit_logplane(all_time[mask] - all_time[mask].min() / (1 + z), 
			log10(wavelengths[passbands] / (1 + z) / 300),
			log10(all_flux[mask] / passband_efficiencies[passbands]))
	else:
		phi_fall, theta_fall, evolfall = numpy.nan, numpy.nan, numpy.nan
	features += [phi_fall, theta_fall, evolfall]

	# plot 
	if len(peakfluxes) > 2:
		peakfluxes = numpy.asarray(peakfluxes)
		Twave, s, slope, chi2diff = bbody_fit(peakfluxes[:,0], peakfluxes[:,1] / numpy.nanmax(peakfluxes[:,1]))
		flux400 = s * Twave**-5 / (numpy.exp(1/(wavenorm/Twave)) - 1)
		features += [Twave, flux400]
	else:
		features += [numpy.nan, numpy.nan]
	
	# measure rising and falling fluxes
	Trise = numpy.median(Trise)
	Tfall = numpy.median(Tfall)
	Tratio = log10(Trise / Tfall)
	features += [Trise, Tfall, Tratio]
	
	# measure whether the SED has peaks and dips
	if len(peakseries) > 0:
		features += [numpy.max(peakseries), numpy.max(dipseries)]
	else:
		features += [0, 0]
	
	#print("   ",len(features))
	fout.write(linefmt % tuple(features))



