# -*- coding:UTF-8 -*-
import numpy as np

np.random.seed(1337)  # for reproducibility
from sklearn.cross_validation import train_test_split
from sklearn.metrics.regression import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from dbn.tensorflow import SupervisedDBNRegression

import matplotlib
matplotlib.use('Agg')
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys
import os
sys.path.append(os.getcwd())

from numpy import exp
from pyspark import *
from dbn import mylogging


def makegraphe(a,b,name_png):
	x = a
	y = b
	plt.scatter(x, y,c='r',alpha=0.5)
	plt.savefig(name_png)  


def main(argv):


########################################################################
	sc = SparkContext("local[16]","stock",batchSize=1000)
	lines = sc.textFile("/data/train_all_shuffle.txt/train_all_shuffle.txt",2)

	print(lines.count() / 400000)
	lines = lines.map(lambda s : np.fromstring(s,dtype=np.float32,sep=","))

	l = []
	num_input=34
	# Training
	regressor = SupervisedDBNRegression(hidden_layers_structure=[200,100],
										learning_rate_rbm=0.1,
										learning_rate=0.01,
										n_epochs_rbm=10,
										n_iter_backprop=200,
										contrastive_divergence_iter=1,
										batch_size=100,
										activation_function='relu',
										optimization_algorithm='sgd',
										l2_regularization=1.0,
										dropout_p=0)
	for x in lines.toLocalIterator():
		l.append(x)
		if(len(l)>200000):
			idata = np.array(l)
		

			X=idata[:,1:num_input+1]
			Y=idata[:,0]

			#X = minmax(X)

			# Splitting data
			X_src_train, X_src_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
			
			X_train   =   X_src_train[:,0:31]#13
			X_test    =   X_src_test[:,0:31]

			# Data scaling,将属性缩放到一个指定范围
			# 常用的方法是将属性缩放到一个指定的最大和最小值（通常是1-0）之间
			# https://www.cnblogs.com/chaosimple/p/4153167.html
			min_max_scaler = MinMaxScaler()
			X_train = min_max_scaler.fit_transform(X_train)
			#################################################################

			# Data scaling,标准化（Z-Score），或者去除均值和方差缩放
			# 公式为：(X-mean)/std  计算时对每个属性/每列分别进行。
			# 将数据按期属性（按列进行）减去其均值，并处以其方差。得到的结果是，对于每个属性/每列来说所有数据都聚集在0附近，方差为1。
			# https://www.cnblogs.com/chaosimple/p/4153167.html
			# standard_scaler = StandardScaler()
			# standard_scaler.fit(X_train)
			# X_train = standard_scaler.transform(X_train)
			#################################################################

			# Train
			##########################################################################################################
			regressor.fit(X_train, Y_train)
			# l = []


	       # Save the model
			regressor.save('model.pkl')

			Y_train_pred = regressor.predict(X_train)

        	# makegraphe(Y_train.reshape(1,len(Y_train))[0],Y_train_pred.reshape(1,len(Y_train_pred))[0],"train_data.png")
			a = Y_train.reshape(1,len(Y_train))[0]
			b = Y_train_pred.reshape(1,len(Y_train_pred))[0]
			plt.scatter(a, b,c='r',alpha=0.5)
			plt.savefig("train_data.png") 
			
			c = np.hstack( (a.reshape(len(a),1),b.reshape(len(a),1)) )
			d = np.hstack( (c,X_src_train[:,31:34] ) )
			np.savetxt("./train_and_predtrain.txt",d)

			print('Done.\nR-squared: %f\nMSE: %f' % (r2_score(Y_train, Y_train_pred), mean_squared_error(Y_train, Y_train_pred)))
	       	##########################################################################################################

	       	# Test
	       	##########################################################################################################
			X_test = min_max_scaler.transform(X_test)
			Y_pred = regressor.predict(X_test)

			# makegraphe(Y_test.reshape(1,len(Y_test))[0],Y_pred.reshape(1,len(Y_pred))[0],"test_data.png")

			a = Y_test.reshape(1,len(Y_test))[0]
			b = Y_pred.reshape(1,len(Y_pred))[0]
			plt.scatter(a, b,c='r',alpha=0.5)
			plt.savefig("test_data.png") 

			c = np.hstack( (a.reshape(len(a),1),b.reshape(len(a),1)) )
			d = np.hstack( (c,X_src_test[:,31:34] ) )
			np.savetxt("./test_and_predtest.txt",d)

			print('Done.\nR-squared: %f\nMSE: %f' % (r2_score(Y_test, Y_pred), mean_squared_error(Y_test, Y_pred)))
			##########################################################################################################
			return


if __name__ == '__main__':
	sys.exit(main(sys.argv))
