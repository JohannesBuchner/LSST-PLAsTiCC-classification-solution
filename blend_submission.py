import sys
import numpy
import pandas

data_input = pandas.read_csv('training_set_metadata.csv')
classes = [6,15,16,42,52,53,62,64,65,67,88,90,92,95,99]
frac_99 = 1.0
N = numpy.array([(data_input.target == cl).sum() for cl in classes], dtype='f')
N[-1] = frac_99
Ntotal = N.sum()

galmask = data_input.hostgal_specz == 0
Ngal = numpy.array([(numpy.logical_and(data_input.target == cl, galmask)).sum() for cl in classes], dtype='f')
Ngal[-1] = frac_99
Ntotal_gal = Ngal.sum()

exgalmask = ~galmask
Nexgal = numpy.array([(numpy.logical_and(data_input.target == cl, exgalmask)).sum() for cl in classes], dtype='f')
Nexgal[-1] = frac_99
Ntotal_exgal = Nexgal.sum()

print('loading prediction to correct, "%s" ...' % (sys.argv[1]))
supp_df = pandas.read_csv('test_set_metadata.csv')
galmask = supp_df.hostgal_specz == 0
df = pandas.read_csv(sys.argv[1])
assert (df.object_id == supp_df.object_id).all(), (df.object_id, supp_df.object_id)
del supp_df
w = 0.0
w_wrongspecz = 0.01
expo = 0.75
expo = 1
#w_wrongspecz = 0

for col, Ni, Ngali, Nexgali in zip(df.columns[1:], N, Ngal, Nexgal):
	print('  adjusting column "%s" ...' % (col))
	prior_gal = Ngali * 1. / Ntotal_gal
	prior_exgal = Nexgali * 1. / Ntotal_exgal
	prior = Ni * 1. / Ntotal
	#prior = w_wrongspecz * prior + (1 - w_wrongspecz) * numpy.where(galmask, prior_gal, prior_exgal)
	df.loc[:,col] = df.loc[:,col]**expo * (1 - w) + prior * w

print("writing data ...")
#data1.values[:,1:] = data1.values[:,1:]**2 * (1 - w) + (N * 1. / Ntotal).reshape((1,-1)) * w
df.to_csv(sys.argv[1] + '_blend_expo%s_outfrac%s_z%s_prior%s.csv.gz' % (expo, frac_99, w_wrongspecz, w), 
	float_format='%.3e', index=False, header=True, compression='gzip', chunksize=100000)


sys.exit(0)


data = """151 6
495 15
924 16
1193 42
183 52
30 53
484 62
102 64
981 65
208 67
370 88
2313 90
239 92
175 95
0.1 99""".split("\n")
data = numpy.array([[float(v) for v in row.split()] for row in data])
N = data[:,0]
Ntotal = N.sum()
df = pandas.read_csv(sys.argv[1])
w = 0.2
for col, Ni in zip(df.columns[1:], N):
	df.loc[:,col] = df.loc[:,col]**0.75 * (1 - w) + (Ni * 1. / Ntotal) * w

#data1.values[:,1:] = data1.values[:,1:]**2 * (1 - w) + (N * 1. / Ntotal).reshape((1,-1)) * w
df.to_csv(sys.argv[1] + '_blend.csv', float_format='%.3e', index=False)


