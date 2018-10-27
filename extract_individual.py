from __future__ import print_function, division
import os, sys
import numpy
from numpy import pi, exp, log10
import pandas
import matplotlib.pyplot as plt
import scipy.stats
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

def fit_logline(time, flux, flux_error):
	x, y, y_err = time, numpy.log10(flux), numpy.log10(flux_error)
	slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
	#excess_variance, concavity = compute_concavity(time, flux, flux_error, ymodel=10**(slope * x + intercept))
	excess_variance, concavity = std_err, r_value

	return [slope, intercept, excess_variance, concavity]

def xyplot(x, y, color=None, lo=None, hi=None):
	if numpy.isfinite(x) and numpy.isfinite(y):
		x = min(hi, max(lo, x))
		y = min(hi, max(lo, y))
		plt.plot(x, y, 'o', color=color)
	elif numpy.isfinite(y):
		y = min(hi, max(lo, y))
		plt.hlines(y, lo, hi, colors=[color])
	elif numpy.isfinite(x):
		x = min(hi, max(lo, x))
		plt.vlines(x, lo, hi, colors=[color])
	plt.xlim(lo, hi)
	plt.ylim(lo, hi)
	

def LC_slopes(prefix, time, flux, flux_error, z):
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
	
	t = numpy.linspace(numpy.min(time_bs)-10, numpy.max(time_bs)+10, 10)
	plt.figure('timeseries-normed' + prefix)
	r = plt.errorbar(x=time_bs - tintercept, y=flux / peakflux, yerr=flux_error / peakflux, linestyle=' ', marker='x')
	color = r[0].get_color()
	if True: #numpy.min(time)-10 < tintercept < numpy.max(time)+10:
		plt.plot(t - tintercept, 10**(lresult[0] * t + lresult[1]) / peakflux, color=color)
		plt.plot(t - tintercept, 10**(rresult[0] * t + rresult[1]) / peakflux, color=color)

	plt.figure('slopes' + prefix)
	#print('slopes: %.4f %.4f' % (lresult[0], rresult[0]))
	xyplot(lresult[0], -rresult[0], color=color, lo=1e-4, hi=0.1)
	
	plt.figure('Lz' + prefix)
	plt.plot(z+1, peakflux, 'x', ms=2, color=color)
	return peakflux, tintercept, lresult[0], rresult[0]

def fit_logplane(x, y, z):
	"""
	minimize perpendicular distance using PCA
	returns angles to xy-plane
	"""
	datamatrix = numpy.log10([x, y, z])
	print(datamatrix.shape, datamatrix)
	# we do a eigenvector search
	U, s, Vh = scipy.linalg.svd(datamatrix)
	# the least important eigenvector points perpendicular
	xf, yf, zf = U[s.argmin()]
	# transfer to spherical coordinates to get angles
	rad = (xf**2+yf**2+zf**2)**0.5
	phi = numpy.arctan2(yf, xf)
	theta = numpy.arccos(zf / rad)
	print('normal vector:', U.shape, s, phi, theta, xf, yf, zf, rad)
	return phi / pi * 180, theta / pi * 180

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
	chi2best = chi2[i]
	best = [Twave_grid[i], s[i]]
	
	return best + [slope, chi2best - chi2_PL]


## ugrizY EBV effect, assuming a flat spectrum
#extinction_factors = [0.0092394833, 0.0336530868, 0.0938432007, 0.1868623254, 0.2921024403, 0.4306847762]
bands = range(6)

# very roughly read off the LSST passband plot
wavelengths = [357, 475, 621, 755, 879, 1000]
passband_efficiencies = [65, 140, 131, 118, 98, 151]

wavelengths = numpy.array(wavelengths)
passband_efficiencies = numpy.array(passband_efficiencies)

print("loading metadata ...")
metadata_all = pandas.read_csv(sys.argv[1] + '_metadata.csv')
metadata_all = metadata_all.set_index('object_id')
print("loading data ...")
object_data_all = pandas.read_csv(sys.argv[1] + '.csv')
object_data_all = object_data_all.set_index('object_id')

if not os.path.exists('viz'): os.mkdir('viz')

for object_id in map(int, sys.argv[2:]):
	metadata = metadata_all.loc[object_id]
	object_data = object_data_all.loc[object_id]
	# columns:
	# columns = ['mjd', 'passband', 'flux', 'flux_err', 'detected']
	# object_columns = ['ra', 'decl', 'gal_l', 'gal_b', 'ddf', 'hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err', 'distmod', 'mwebv', 'target']

	fileprefix = 'viz/object%d_' % (object_id)


	specz = metadata['hostgal_specz']
	photoz = metadata['hostgal_photoz']
	photoz_error = metadata['hostgal_photoz_err']
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
	is_detected = object_data['detected'].values == 1
	print(object_id, len(object_data), is_detected.sum())
	if not numpy.logical_and(all_flux > 0, is_detected).any():
		print("%d NEVER DETECTED!" % object_id)
		continue

	plt.figure('SEDevol', figsize=(5, 20))
	plt.figure('temperature-timeseries', figsize=(5, 10))
	peakfluxes = []
	peaktimes = []
	ndips = []
	npeaks = []
	runstat2 = [numpy.nan]*7

	for passband in bands:
		prefix = '%d' % passband
		fileprefix = 'viz/object%d_%d' % (object_id, passband)

		mask = numpy.logical_and(all_passband == passband, 
			numpy.logical_and(all_flux > 0, is_detected))
		print("  band %d: %d data points" % (passband, mask.sum()))
		if mask.sum() < 2: continue
		
		flux = all_flux[mask]
		flux_error = all_flux_error[mask]
		time = all_time[mask]
		# count dips: where the flux is lower than earlier and later
		mask_left_down  = flux[1:-1] + flux_error[1:-1] < (flux[0:-2] - flux_error[0:-2]) * 0.8
		mask_right_up   = flux[1:-1] + flux_error[1:-1] < (flux[2:]   - flux_error[2:]  ) * 0.8
		ndips.append(numpy.logical_and(mask_left_down, mask_right_up).mean())
		mask_left_up    = flux[1:-1] - flux_error[1:-1] > (flux[0:-2] + flux_error[0:-2]) / 0.8
		mask_right_down = flux[1:-1] - flux_error[1:-1] > (flux[2:]   + flux_error[2:]  ) / 0.8
		#print("  ", passband, mask.sum(), 
		#	numpy.logical_and(mask_left_up, mask_right_down).sum(), 
		#	numpy.logical_and(mask_left_down, mask_right_up).sum())
		npeaks.append(numpy.logical_and(mask_left_up, mask_right_down).mean())
		if passband == 2:
			mask_down = flux[1:] - flux_error[1:] < flux[:-1] + flux_error[:-1]
			mask_up   = flux[1:] + flux_error[1:] > flux[:-1] - flux_error[:-1]
			runlengths_up   = runs_of_ones(mask_up)
			runlengths_down = runs_of_ones(mask_down)
			print("  runs:", runlengths_up, runlengths_down)
			runstat2 = [
				len(flux),
				numpy.logical_and(mask_left_down, mask_right_up).sum(),
				numpy.logical_and(mask_left_up, mask_right_down).sum(),
				numpy.mean(runlengths_up), numpy.std(runlengths_up), 
				numpy.mean(runlengths_down), numpy.std(runlengths_down), 
				]
			plt.figure("dist")
			plt.hist(numpy.log10(flux / numpy.median(flux)), cumulative=True, bins=100, density=True, histtype='step')
			plt.xlabel('Flux')
			plt.savefig(fileprefix + 'dist.pdf', bbox_inches='tight')
			plt.close()
			
		
		# create features
		peakflux, tpeak, lslope, rslope = LC_slopes('%d' % passband, time, flux, flux_error, z)
		peaktimes.append(tpeak)
		#print('    peak@%d' % tpeak)
		peakfluxes.append([wavelengths[passband] / (1 + z), peakflux / passband_efficiencies[passband]])
		
		plt.figure('timeseries%d' % passband)
		r = plt.errorbar(x=(time - all_time.min()) / (1 + z), y=flux / flux.max(), yerr=flux_error / flux.max(), linestyle='-', marker='x')
		plt.xlabel('Restframe time [d]')
		plt.ylabel('Flux')
		plt.yscale('log')
		plt.savefig(fileprefix + 'timeseries.pdf',  bbox_inches='tight')
		plt.close()

		plt.figure('timeseries-normed' + prefix)
		plt.xlabel('Restframe time [d]')
		plt.ylabel('Flux')
		plt.yscale('log')
		plt.ylim(1e-3, 1)
		plt.savefig(fileprefix + 'timeseries-normed.pdf',  bbox_inches='tight')
		plt.close()

		plt.figure('slopes' + prefix)
		plt.xlabel('Rising slope')
		plt.ylabel('Falling slope')
		plt.plot([1e-4,0.1], [1e-4,0.1], ':', color='k')
		plt.xscale('log')
		plt.yscale('log')
		plt.savefig(fileprefix + 'slopes.pdf', bbox_inches='tight')
		plt.close()

		plt.figure('Lz' + prefix)
		plt.xlabel('1+z')
		plt.ylabel('Peak Flux')
		plt.yscale('log')
		plt.savefig(fileprefix + 'Lz.pdf', bbox_inches='tight')
		plt.close()

	fileprefix = 'viz/object%d_' % (object_id)
	if peaktimes:
		tpeak_lo, tpeak_hi, tpeak = min(peaktimes), max(peaktimes), numpy.median(peaktimes)
	else:
		tpeak_lo, tpeak_hi, tpeak = None, None, all_time[all_flux.argmax()] / (1 + z)
	Trise = []
	Tfall = []
	lastt = -100
	deltat = 0.1
	blueratioseries = []
	redratioseries = []
	Tseries = []
	for t in numpy.unique(all_time[numpy.logical_and(all_flux > 0, is_detected)]):
		#print('deltat: %.2f' % (t - lastt))
		if t < lastt + deltat:
			continue
		lastt = t
		trest = t / (1 + z)
		mask = numpy.logical_and(numpy.logical_and(all_time >= t, all_time < t + deltat), 
			numpy.logical_and(all_flux > 0, is_detected))
		if (all_passband[mask] == 0).any() and (all_passband[mask] == 1).any():
			blueratioseries.append([t, float(all_flux[mask][all_passband[mask] == 0].mean() / all_flux[mask][all_passband[mask] == 1].mean())])
		if (all_passband[mask] == 4).any() and (all_passband[mask] == 5).any():
			redratioseries.append([t, float(all_flux[mask][all_passband[mask] == 4].mean() / all_flux[mask][all_passband[mask] == 5].mean())])
		
		if mask.sum() > 3:
			passband = all_passband[mask]
			Twave, s, slope, chi2diff = bbody_fit(wavelengths[passband] / (1 + z), all_flux[mask] / all_flux[mask].max() / passband_efficiencies[passband])

			plt.figure('SEDevol')
			norm = 10**((trest - tpeak)/20.)
			#if norm > 1000 or norm < 0.001:
			#	norm = numpy.nan
			r = plt.plot(wavelengths[passband] / (1 + z), norm * all_flux[mask] / all_flux[mask].max() / passband_efficiencies[passband], 'o ')
			color = r[0].get_color()
			Twave, s, slope, chi2diff = bbody_fit(wavelengths[passband] / (1 + z), all_flux[mask] / all_flux[mask].max() / passband_efficiencies[passband])
			wave = numpy.linspace(Tmin*3, Tmax/3, 400)
			plt.plot(wave, norm * s * wave**-5 / (numpy.exp(1/(wave/Twave)) - 1), ':', color=color)
			
			plt.figure('color')
			wavenorm = 400 # evaluate the SED at 1um rest-frame
			flux400 = s * Twave**-5 / (numpy.exp(1/(wavenorm/Twave)) - 1)
			plt.plot(Twave, flux400, 'o', ms=2, color=color)
			
			Tseries.append([trest, Twave, slope, chi2diff])
			if tpeak_lo is not None and trest < tpeak_lo:
				Trise.append(Twave)
			if tpeak_hi is not None and trest > tpeak_hi:
				Tfall.append(Twave)
	if len(Tseries) > 2:
		plt.figure('SEDevol')
		plt.xlabel('Restframe Wavelength [nm]')
		plt.ylabel('Flux [offset by time]')
		plt.yscale('log')
		plt.xscale('log')
		plt.savefig(fileprefix + 'SEDevol.pdf', bbox_inches='tight')
		plt.close()

		Tseries = numpy.array(Tseries)
		plt.figure('temperature-timeseries')
		plt.subplot(3, 1, 1)
		plt.plot(Tseries[:,0] - tpeak, Tseries[:,1], 'x-', ms=2)
		plt.subplot(3, 1, 2)
		plt.plot(Tseries[:,0] - tpeak, Tseries[:,2], 'x-', ms=2)
		plt.subplot(3, 1, 3)
		plt.plot(Tseries[:,0] - tpeak, Tseries[:,3], 'x-', ms=2)
		plt.subplot(3, 1, 1)
		plt.ylabel('Temperature [nm]')
		plt.xlabel('Restframe time [d]')
		plt.yscale('log')
		plt.ylim(Tmin, Tmax)
		plt.subplot(3, 1, 2)
		plt.ylabel('SED slope')
		plt.xlabel('Restframe time [d]')
		plt.subplot(3, 1, 3)
		plt.ylabel('$\chi^2$(BB)-$\chi^2$(PL)')
		plt.xlabel('Restframe time [d]')
		plt.savefig(fileprefix + 'Tseries.pdf', bbox_inches='tight')
		plt.close()

	if len(redratioseries) > 1:
		tred, redratio = numpy.transpose(redratioseries)
		redrise, redfall = numpy.mean(redratio[tred < tpeak]), numpy.mean(redratio[tred > tpeak])
		print("red ratios: %.2f %.2f" % (redrise, redfall))
	if len(blueratioseries) > 1:
		tblue, blueratio = numpy.transpose(blueratioseries)
		bluerise, bluefall = numpy.mean(blueratio[tblue < tpeak]), numpy.mean(blueratio[tblue > tpeak])
		print("blue ratios: %.2f %.2f" % (bluerise, bluefall))

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
	mask = numpy.logical_and(numpy.logical_and(all_flux > 0, is_detected), all_time / (1 + z) > tpeak)
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
	plt.figure('slopeevol')
	r = plt.plot([phi_rise, -phi_fall], [theta_rise, theta_fall], 'o-')
	color = r[0].get_color()
	plt.xlabel(r'Flux-Time slope $\log\phi = t \times \tau$')
	plt.ylabel(r'Flux-wavelength slope $\log\phi=\log(\lambda/10\mu m) \times \Lambda$')
	plt.xlim(-0.1, 0.2)
	plt.vlines(0, -3, 5, linestyle=':', colors='k')
	plt.ylim(-3, 5)
	plt.savefig(fileprefix + 'slopeevol.pdf', bbox_inches='tight')
	plt.close()
	
	plt.figure('colorevol')
	xyplot(evolrise, evolfall, color=color, lo=-0.5, hi=0.5)
	plt.xlabel(r'Rise evolution')
	plt.ylabel(r'Fall evolution')
	plt.savefig(fileprefix + 'colorevol.pdf', bbox_inches='tight')
	plt.close()

	# plot 
	if len(peakfluxes) > 0:
		peakfluxes = numpy.asarray(peakfluxes)
		plt.figure('SED')
		plt.plot(peakfluxes[:,0], peakfluxes[:,1] / numpy.nanmax(peakfluxes[:,1]), 'o-')
		Twave, s, slope, chi2diff = bbody_fit(peakfluxes[:,0], peakfluxes[:,1] / numpy.nanmax(peakfluxes[:,1]))
		wave = numpy.linspace(Tmin*3, Tmax/3, 400)
		plt.plot(wave, s * wave**-5 / (numpy.exp(1/(wave/Twave)) - 1), color=color)
		plt.xlabel('Restframe Wavelength [nm]')
		plt.ylabel('Flux')
		plt.yscale('log')
		plt.xscale('log')
		plt.savefig(fileprefix + 'SED.pdf', bbox_inches='tight')
		plt.close()
		
		plt.figure('color')
		wavenorm = 400 # evaluate the SED at 1um rest-frame
		flux400 = s * Twave**-5 / (numpy.exp(1/(wavenorm/Twave)) - 1)
		plt.plot(Twave, flux400, 'x', ms=2, color=color)
		plt.xlabel('Temperature [nm]')
		plt.ylabel('Flux')
		plt.xlim(Tmin, Tmax)
		plt.yscale('log')
		plt.savefig(fileprefix + 'color.pdf', bbox_inches='tight')
		plt.close()

	# measure rising and falling fluxes
	Trise = numpy.median(Trise)
	Tfall = numpy.median(Tfall)
	plt.figure('slopecolor')
	xyplot(Trise, Tfall, color=color, lo=300, hi=40000)
	plt.xlabel('Rise Temperature [nm]')
	plt.ylabel('Fall Temperature [nm]')
	plt.xlim(Tmin, Tmax)
	plt.ylim(Tmin, Tmax)
	plt.savefig(fileprefix + 'slopecolor.pdf', bbox_inches='tight')
	plt.close()
	
	if len(ndips) > 0:
		print("# of dips:", numpy.max(ndips), "# of peaks:", numpy.max(npeaks))
	
	plt.figure('dipstat')
	plt.plot(range(7), runstat2, marker='s', color=color)
	plt.xticks(range(7), '#data,#dips,#peaks,avg\nuprun,std\nuprun,avg\ndownrun,std\ndownrun'.split(','))
	plt.savefig(fileprefix + 'dipstat.pdf', bbox_inches='tight')
	plt.close()
	plt.close('all')
	#[numpy.max(ndips) if ndips else numpy.nan, 
	#numpy.max(npeaks) if npeaks else numpy.nan], 


