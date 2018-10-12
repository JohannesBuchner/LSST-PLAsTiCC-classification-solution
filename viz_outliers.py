from __future__ import print_function, division
from alltrain import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

X_galmask = train.hostgal_photoz.values == 0
X = mytransformer.fit_transform(X)
print('gal/exgal:', X_galmask.sum(), (~X_galmask).sum())
#label_colors = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])

execute = unknown_data_file is not None
unknown_galmask = None
unknown = None
if execute:
	unknown = pandas.read_csv(unknown_data_file)
	unknown.pop('object_id')        
	unknown_galmask = unknown.hostgal_photoz.values == 0
	print('gal/exgal:', unknown_galmask.sum(), (~unknown_galmask).sum())
	unknown = unknown.values
	print('unknown:', unknown.shape)
	unknown[~numpy.isfinite(unknown)] = -99
	unknown = mytransformer.transform(unknown)

def make_transform(name, tsne, X, Y, Z):
	print()
	print("inputs:", X.shape, Y.shape, numpy.shape(Z))
	sys.stdout.write('running %s ...\r' % name)
	sys.stdout.flush()
	t0 = time()
	# if Z is given, we should merge together X+Z and transform with them
	# then color only the labelled (X,Y) values
	if Z is None:
		XZ = X
		YY = Y
	else:
		XZ = numpy.vstack((X, Z))
		YY = numpy.hstack((Y, numpy.zeros(len(Z))-1 ))
	L = tsne.fit_transform(XZ)
	print('running %s complete (%.2fs)' % (name, time() - t0))
	# plot results
	plt.figure(figsize=(10,10))
	# give each class a different color
	labels_here = numpy.unique(Y)
	for cls in labels_here:
		print("  plotting class %d" % cls)
		mask_cls = YY == cls
		plt.scatter(L[mask_cls,0], L[mask_cls,1], label='class %d' % cls)
	
	plt.title(name)
	mask_cls = YY == -1
	plt.scatter(L[mask_cls,0], L[mask_cls,1], c='gray', marker='x', s=2, label='unknown')
	print("saving plot...")
	plt.legend(loc='best', prop=dict(size=8))
	plt.savefig('viz_%s.pdf' % name)
	plt.close()

def train_and_evaluate(name, tsne):
	make_transform(name + '_exgal', tsne, X_white[~X_galmask,:], Y[~X_galmask], 
		unknown_white[~unknown_galmask,:] if unknown is not None else None)
	make_transform(name + '_gal', tsne, X_white[X_galmask,:], Y[X_galmask], 
		unknown_white[unknown_galmask,:] if unknown is not None else None)
	#make_transform(name + '_all', tsne, X_white, Y, unknown_white)
	return tsne

n_components = int(os.environ.get('NPCACOMP', '40'))
perplexity = int(os.environ.get('PERPLEXITY', '30'))

prefix = 'PCA-%d' % n_components
print("dimensionality reduction with %s ..." % prefix)
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X)
print("done in %0.3fs" % (time() - t0))
X_white = pca.transform(X)
if execute:
	unknown_white = pca.transform(unknown)

train_and_evaluate(prefix + 'TSNE%d' % perplexity, tsne = TSNE(perplexity=perplexity, n_iter=5000))



