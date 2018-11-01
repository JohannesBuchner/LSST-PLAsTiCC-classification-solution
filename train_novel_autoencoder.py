from __future__ import print_function, division
from alltrain import *
column_names = train.columns[valid_column_mask]
del train
import matplotlib.pyplot as plt

X = mytransformer.fit_transform(X)

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
	unknown = mytransformer.transform(unknown)

import keras                   
from keras.models import Model, load_model                             
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn.model_selection import train_test_split 

input_dim = X.shape[1]
# how many neurons for compressed representation?

mse_limit = 10.0

for encoding_dim in 2, : #3, 4, 5:
	name = 'AE%d' % encoding_dim
	prefix = ('SIMPLE' if simplify_space else '') + transform + '-AE%d-' % encoding_dim
	
	input_layer = Input(shape=(input_dim, ))
	encoder = Dense(int(input_dim / 2), activation="tanh",activity_regularizer=regularizers.l1(10e-5))(input_layer)
	encoder = Dense(int(encoding_dim * 1.5), activation="tanh")(encoder)
	encoder = Dense(encoding_dim, activation="tanh")(encoder)
	decoder = Dense(int(encoding_dim * 1.5), activation='tanh')(encoder)
	decoder = Dense(int(input_dim / 2), activation='tanh')(decoder)
	decoder = Dense(input_dim, activation='tanh')(decoder)
	autoencoder = Model(inputs=input_layer, outputs=decoder)
	autoencoder.summary()

	nb_epoch = 100
	batch_size = 50
	autoencoder.compile(optimizer='adam', loss='mse' )

	t0 = time()
	history = autoencoder.fit(X, X, 
		epochs=nb_epoch, batch_size=batch_size,
		shuffle=True, validation_split=0.1, verbose=0)

	print('Time to run the model: %.1fs.' % (time() - t0))

	plt.plot(history.history['loss'], label='loss')
	plt.plot(history.history['val_loss'], label='validation loss')
	plt.xlabel('Iteration')
	plt.ylabel('Loss')
	plt.legend(loc='best')
	plt.savefig('ae_train_%d.pdf' % encoding_dim, bbox_inches='tight')
	plt.close()
	
	mse = ((X - autoencoder.predict(X))**2).mean(axis=1)
	
	for label in labels:
		print('class %s: loss: %.3f' % (label, mse[Y == label].mean()))
	plt.plot(mse, '.', ms=2)
	plt.xlabel('Object')
	plt.yscale('log')
	plt.ylabel('Loss')
	plt.savefig('ae_loss_%d.pdf' % encoding_dim, bbox_inches='tight')
	plt.close()
	
	if execute:
		print('predictions for unknown data...')
		t0 = time()
		prediction = autoencoder.predict(unknown)
		mse = ((unknown - prediction)**2).mean(axis=1)
		for i in numpy.argsort(mse)[::-1]:
			if mse[i] < mse_limit: break
			print('outlier %d: mse=%f' % (unknown_object_ids[i], mse[i]))
			# find the columns that contribute more than 1% of the loss
			mask = ((unknown[i,:] - prediction[i,:])**2) > 0.01 * ((unknown[i,:] - prediction[i,:])**2).sum()
			for j, col in enumerate(column_names):
				if not mask[j]: continue
				print("   column %s: have %s  predicted: %s" % (col, unknown[i,j], prediction[i,j]))
		outlier_prob = mse / numpy.median(mse) * 0.001
		outlier_prob[outlier_prob > 1] = 1.0
		
		plt.plot(mse, '.', ms=2)
		plt.xlabel('Object')
		plt.yscale('log')
		plt.ylabel('Loss')
		plt.savefig('ae_loss_predict_%d.pdf' % encoding_dim, bbox_inches='tight')
		plt.close()
		i = numpy.where(mse > mse_limit)[0]
		print('novel: %d/%d (%.2f%%)' % (len(i), len(unknown), len(i) * 100. / len(unknown)))
		numpy.savetxt(unknown_data_file + '_novel_%s.csv' % name, numpy.transpose([i, unknown_object_ids[i], mse[i]]), delimiter=',', fmt='%d,%d,%f')
		numpy.savetxt(unknown_data_file + '_novel_%s_prob.csv' % name, outlier_prob, delimiter=',', fmt='%.4e')
		
		print('predictions done after %.1fs' % (time() - t0))
	

