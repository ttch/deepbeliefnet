# -*- coding:UTF-8 -*-
# select parameters to output DBN model
import numpy as np

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_boston
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
#from dbn import mylogging
import json



sc = SparkContext("local[16]","stock",batchSize=1000)




def makegraphe(a,b,name_png):
	x = a
	y = b
	plt.scatter(x, y,c='r',alpha=0.5)
	plt.savefig(name_png)  

def readargs(i):
	f = open("args.cfg","r")
	ss = ""
	ii = 0
	for x in f.readlines():
		ss = x
		ss = ss.replace("\n","")
		if ss == "" :
			continue
		if int(ii) == int(i):
			print(ii,i,"---------------")
			break
		ii = ii + 1
	f.close()
	print(ss)
	return json.loads(ss)

def main(argv):
	argno = argv[1]
	print("_----------------------------:",argv[1])
	__args = readargs(argno)
	print(__args)

	(tr,ts,testr,tests) = train_args(__args["hidden_layers_structure"],
		__args["learning_rate_rbm"],
		__args["learning_rate"],
		__args["n_epochs_rbm"],
		__args["n_iter_backprop"],
		__args["batch_size"],
		__args["activation_function"],
		__args["dropout_p"],
		argno
		)
	
	writelog("{} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(__args["i"] \
		,tr,ts,testr,tests,__args["hidden_layers_structure"], \
		__args["learning_rate_rbm"], \
		__args["learning_rate"], \
		__args["n_epochs_rbm"], \
		__args["n_iter_backprop"], \
		__args["batch_size"], \
		__args["activation_function"], \
		__args["dropout_p"] \
		))

def writelog(context):
	f = open("result.log","a+")
	f.write(context)
	f.close()







def train_args(hidden_layers_structure,learning_rate_rbm,learning_rate,n_epochs_rbm,
	n_iter_backprop,batch_size,activation_function,dropout_p,argno):
########################################################################
	global sc
	lines = sc.textFile("/data",2)
	print(lines.count())
	lines = lines.map(lambda s : np.fromstring(s,dtype=np.float32,sep=" "))

	l = []
	total_num_input=18
	# Training
	regressor = SupervisedDBNRegression(hidden_layers_structure=hidden_layers_structure,
										learning_rate_rbm=learning_rate_rbm,
										learning_rate=learning_rate,
										n_epochs_rbm=n_epochs_rbm,
										n_iter_backprop=n_iter_backprop,
										batch_size=batch_size,
										activation_function=activation_function,
										dropout_p=dropout_p)
	for x in lines.toLocalIterator():
		l.append(x)
		if(len(l)>=28881):
			idata = np.array(l)
		

			X=idata[:,1:total_num_input+1]
			Y=idata[:,0]

			#X = minmax(X)

			# Splitting data
			X_train = X[0:19000,:]
			X_test = X[19000:28881,:]
			Y_train = Y[0:19000]
			Y_test = Y[19000:28881]




			# Train
			X_train_select = X_train[:,0:total_num_input]
			regressor.fit(X_train_select, Y_train)
			
                        # Save the model
			regressor.save('./out/model_{}.pkl'.format(argno))
			# regressor.save('model.pkl')

			Y_train_pred = regressor.predict(X_train_select)
			# l = []


			print('Done.\nR-squared: %f\nMSE: %f' % (r2_score(Y_train, Y_train_pred), mean_squared_error(Y_train, Y_train_pred)))
	       	        ##########################################################################################################

	       	        # Test
	       	        ##########################################################################################################
			X_test_select = X_test[:,0:total_num_input]
			Y_test_pred = regressor.predict(X_test_select)

 
			print('Done.\nR-squared: %f\nMSE: %f' % (r2_score(Y_test, Y_test_pred), mean_squared_error(Y_test, Y_test_pred)))
			##########################################################################################################
			return(r2_score(Y_train, Y_train_pred), mean_squared_error(Y_train, Y_train_pred),
				r2_score(Y_test, Y_test_pred), mean_squared_error(Y_test, Y_test_pred))


if __name__ == '__main__':
	sys.exit(main(sys.argv))
