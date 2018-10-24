from __future__ import print_function, division
from alltrain import *
import matplotlib.pyplot as plt

mask = train.hostgal_specz > 0

specz = train.hostgal_specz[mask].values
photoz = train.hostgal_photoz[mask].values

Nz = 15
zindex = (numpy.log10(photoz+1) * Nz / 0.5).astype(int)
zindex[zindex <  0] = 0
zindex[zindex > Nz-1] = Nz-1
print(numpy.unique(zindex))
print(photoz[zindex == 0])
zdists = [[] for i in range(Nz)]
for zi, speczi in zip(zindex, specz):
	zdists[zi].append(speczi)

#zgrid = numpy.logspace(0, 0.5, 20)
#for zlo, zhi in zip(zgrid[:-1], zgrid[1:]):
#	mask = numpy.logical_and(photoz >= zlo, photoz < zhi)
#	print(zlo, zhi, mask.sum())
#	plt.hist(specz[mask], bins=100, cumulative=True, density=True, histtype='step')
for zdist in zdists:
	print(len(zdist))
	plt.hist(zdist, bins=100, cumulative=True, density=True, histtype='step')

plt.xlabel('specz')
plt.savefig('zdist.pdf', bbox_inches='tight')
plt.close()

