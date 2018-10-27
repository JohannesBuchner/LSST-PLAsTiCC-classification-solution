from __future__ import print_function, division
from alltrain import *
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

columns = train.columns[valid_column_mask]
del train

execute = unknown_data_file is not None
if execute:
	print('reading data file to predict ...')
	unknown = pandas.read_csv(unknown_data_file)
	unknown_object_ids = unknown.pop('object_id').values
	unknown = unknown.values                  
	print('unknown:', unknown.shape)
	if simplify_space:
		unknown = unknown[:,column_mask]
	unknown = imp.transform(unknown)

for i, colname in enumerate(columns):
	for j, colname2 in enumerate(columns[:i]):
		r_value, p_value = scipy.stats.spearmanr(X[:,i], X[:,j], nan_policy='omit')
		if abs(r_value) < 0.5: continue
		print("%-15s %-15s: %.3f" % (colname, colname2, r_value))

colgroup_masks = {}
for i, colname in enumerate(columns):
	Xcol = X[:,i]
	Xmean, Xstd = Xcol.mean(), Xcol.std()
	Xlo, Xmed, Xhi = scipy.stats.mstats.mquantiles(Xcol, [0.01, 0.5, 0.99])
	if execute:
		Ycol = unknown[:,i]
		Ymean, Ystd = Ycol.mean(), Ycol.std()
		Ylo, Ymed, Yhi = scipy.stats.mstats.mquantiles(Ycol, [0.01, 0.5, 0.99])
	print("%s:" % colname)
	Xout = ' '.join(['%d' % numpy.logical_or(Xcol > Xmed + (Xhi - Xmed) * i, Xcol < Xmed - (Xmed - Xlo) * i).sum() for i in range(2, 5)])
	if execute:
		Yout = ' '.join(['%d' % numpy.logical_or(Ycol > Xmed + (Xhi - Xmed) * i, Ycol < Xmed - (Xmed - Xlo) * i).sum() for i in range(2, 5)])
		
	print("    %s +- %s  [%s .. %s] (train) \t%s" % (Xmean, Xstd, Xlo, Xhi, Xout))
	if execute:
		print("    %s +- %s  [%s .. %s] (test). \t%s" % (Ymean, Ystd, Ylo, Yhi, Yout))
	
	colgroup = colname.split('_')[-1]
	for label in encoder.classes_:
		mask = Y == label
		if mask.any():
			plt.hist(Xcol[mask], cumulative=True, bins=1000, density=True,
				histtype='step', label='Class %d' % label)
	if execute:
		plt.hist(Ycol, cumulative=True, bins=1000, density=True,
			histtype='step', label='unknown')
		plt.xlim(max(Ycol.min(), Xmed - (Xhi - Xmed)*3), min(Ycol.max(), Xmed + (Xhi - Xmed)*3))
	plt.xlabel(colname)
	plt.ylabel("Cumulative fraction")
	plt.legend(loc='best')
	plt.savefig(unknown_data_file + '_novel_col%s.pdf' % colname, bbox_inches='tight')
	plt.close()
	
	if execute:
		i = 2
		mask = numpy.logical_or(Ycol > Xmed + (Xhi - Xmed) * i, Ycol < Xmed - (Xmed - Xlo) * i)
		mask[Ycol.argmax()] = True
		orig_mask = colgroup_masks.get(colgroup)
		if orig_mask is not None:
			mask = numpy.logical_or(mask, orig_mask)
		colgroup_masks[colgroup] = mask

for colgroup, mask in colgroup_masks.items():
	print('writing %s: %d items' % (colgroup, mask.sum()))
	i = numpy.where(mask)[0]
	numpy.savetxt(unknown_data_file + '_novel_col%s.csv' % colgroup, numpy.transpose([i, unknown_object_ids[i]]), delimiter=',', fmt='%d')


