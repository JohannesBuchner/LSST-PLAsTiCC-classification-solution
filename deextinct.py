import numpy
import matplotlib.pyplot as plt
import extinction

for filtername in 'ugrizy':
	wave_nm, filter_throughput = numpy.loadtxt('throughputs/baseline/filter_%s.dat' % filtername).transpose()

	wave_aa = wave_nm * 10

	# assume spectrum
	flux_input = filter_throughput*0 + 1

	filter_flux = numpy.trapz(y=flux_input * filter_throughput, x=wave_nm)

	r_v = 3.1
	ebv_values = numpy.logspace(-2, 0.3, 10)
	flux_values = []
	#r_v = a_v / EBV
	for ebv in ebv_values:
		a_v = r_v * ebv
		extinction_throughput = extinction.apply(extinction.calzetti00(wave_aa, a_v, r_v, unit='aa', out=None), flux_input)
		filter_extincted_flux = numpy.trapz(y=extinction_throughput * filter_throughput, x=wave_nm) / filter_flux
		flux_values.append(filter_extincted_flux)
	plt.plot(ebv_values, flux_values, label=filtername)
	
	ebv = 1.
	a_v = r_v * ebv
	extinction_throughput = extinction.apply(extinction.calzetti00(wave_aa, a_v, r_v, unit='aa', out=None), flux_input)
	filter_extincted_flux = numpy.trapz(y=extinction_throughput * filter_throughput, x=wave_nm) / filter_flux
	print('%s_factor = %.10f' % (filtername, filter_extincted_flux))

plt.legend(loc='best')
plt.yscale('log')
plt.ylabel('Flux')
plt.xlabel('$A_V$')
plt.savefig('deextinct.pdf', bbox_inches='tight')
plt.close()

