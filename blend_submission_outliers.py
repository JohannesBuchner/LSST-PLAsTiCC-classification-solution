"""
Combine classifier predictions with outlier prediction

Parameters:

PRIOR_STRENGTH		weight of prior
PRIOR_FLATNESS		prior flatness (1: scale with training distribution, 0: flat over classes)
SPECZERR		fraction of wrong gal/exgal classification
EXPO			exponent on classifier probability
OUTLIER_CONF		strength of outlier votes


"""

import sys, os
import numpy
import pandas

flatness = float(os.environ.get('PRIOR_FLATNESS', '1.0'))

data_input = pandas.read_csv('training_set_metadata.csv')
classes = [6,15,16,42,52,53,62,64,65,67,88,90,92,95,99]
N = numpy.array([(data_input.target == cl).sum() for cl in classes], dtype='f')
N = N**(flatness + 1e-30)
N[-1] = N[N>0].min()
Ntotal = N.sum()

galmask = data_input.hostgal_specz == 0
Ngal = numpy.array([(numpy.logical_and(data_input.target == cl, galmask)).sum() for cl in classes], dtype='f')
print("gal   classes:  ", numpy.asarray(classes)[Ngal>0], numpy.where(Ngal>0))
Ngal = Ngal**(flatness + 1e-30)
Ngal[-1] = Ngal[Ngal>0].min()
Ntotal_gal = Ngal.sum()

exgalmask = ~galmask
Nexgal = numpy.array([(numpy.logical_and(data_input.target == cl, exgalmask)).sum() for cl in classes], dtype='f')
print("exgal classes:  ", numpy.asarray(classes)[Nexgal>0], numpy.where(Nexgal>0))
Nexgal = Nexgal**(flatness + 1e-30)
Nexgal[-1] = Nexgal[Nexgal>0].min()
Ntotal_exgal = Nexgal.sum()

print("priors:")
print("gal:  ", Ngal / Ntotal_gal)
print("exgal:", Nexgal / Ntotal_exgal)
print("all:  ", N / Ntotal)
print()


print('loading prediction to correct, "%s" ...' % (sys.argv[1]))
supp_df = pandas.read_csv('test_set_metadata.csv')
galmask = supp_df.hostgal_photoz == 0
df = pandas.read_csv(sys.argv[1])
assert (df.object_id == supp_df.object_id).all(), (df.object_id, supp_df.object_id)
del supp_df
w = float(os.environ.get('PRIOR_STRENGTH', '0.0'))
w_wrongspecz = float(os.environ.get('SPECZERR', '0.01'))
expo = float(os.environ.get('EXPO', '1.0'))

outlier_voteweights = {
	'EllEnvelope':1, 
	'IsolForest':1,
}
outlier_confidence = float(os.environ.get('OUTLIER_CONF', '1.0'))

outlier_votes = 0
nvotes_total = 0
for outliertechnique in 'EllEnvelope', 'IsolForest':
	for thresh in 0.001, 0.01, 0.04:
		print('loading outlier votes of %s[%s] ...' % (outliertechnique, thresh))
		is_outlier = -1 == numpy.loadtxt('test_set_all_sorted.csv.gz_novel_%s-%s.csv' % (outliertechnique, thresh))
		outlier_votes = outlier_votes + is_outlier * outlier_voteweights[outliertechnique]
		nvotes_total += 1


for col, Ni, Ngali, Nexgali in zip(df.columns[1:], N, Ngal, Nexgal):
	print('  adjusting column "%s" ...' % (col))
	if col == 'class_99':
		# novelty class: the other probabilities should sum to 1
		predictor = outlier_votes * outlier_confidence / nvotes_total
	else:
		predictor = df.loc[:,col]**expo
	
	prior_gal = Ngali * 1. / Ntotal_gal
	prior_exgal = Nexgali * 1. / Ntotal_exgal
	prior = Ni * 1. / Ntotal
	
	# blend in the prior
	prior = w_wrongspecz * numpy.where(~galmask, prior_gal, prior_exgal) + (1 - w_wrongspecz) * numpy.where(galmask, prior_gal, prior_exgal)
	
	df.loc[:,col] = predictor * (1 - w) + prior * w

print("writing data ...")
#data1.values[:,1:] = data1.values[:,1:]**2 * (1 - w) + (N * 1. / Ntotal).reshape((1,-1)) * w
df.to_csv(sys.argv[1] + '_blend_expo%s_noveltyconf%s_z%s_prior%sflat%s.csv.gz' % (expo, outlier_confidence, w_wrongspecz, w, flatness), 
	float_format='%.3e', index=False, header=True, compression='gzip', chunksize=100000)


