# -*- coding:UTF-8 -*-
# recon data combine by built model
import numpy as np

np.random.seed(1337)  # for reproducibility
from sklearn.cross_validation import train_test_split
from sklearn.metrics.regression import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from dbn.tensorflow import SupervisedDBNRegression
from dbn import utils
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




def bg_split(X):
	xl = X.shape[0] / 10000
	fen = xl / 40
	xs = np.vsplit(X,fen)
	return xs

def test(X):
	regressor = SupervisedDBNRegression.load('./model/model_109.pkl')
	Y_pred = regressor.predict(X)
	return Y_pred
def main(argv):


########################################################################
	sc = SparkContext("local[16]","stock",batchSize=1000)
	lines = sc.textFile("/data",2)

	print(lines.count())
	lines = lines.map(lambda s : np.fromstring(s,dtype=np.float32,sep=" "))

	l = []
	num_input=18

	for x in lines.toLocalIterator():
		l.append(x)

	idata = np.array(l)
	X = idata[:,1:num_input+1]
	Y_true = idata[:,0]
	xs = bg_split(X)


	Y_pred = np.vstack( [ test(x) for x in xs] )


	a = Y_true.reshape(1,len(Y_true))[0]
	b = Y_pred.reshape(1,len(Y_pred))[0]
				
	plt.scatter(a, b,c='r',alpha=0.4)
	plt.savefig("./output/validate_data.png") 
			
	print('Done.\nR-squared: %f\nMSE: %f' % (r2_score(Y_true, Y_pred), mean_squared_error(Y_true, Y_pred)))		
	
	c = np.hstack( (a.reshape(len(a),1),b.reshape(len(a),1)) )
	d = np.hstack( (c, idata[:,1:21] ) )
	#e = np.hstack( (d, idata[:,20:22] ) )
	np.savetxt('./output/recon_data.txt',d)

	print('Done! save result!')

	return 0

if __name__ == '__main__':
	sys.exit(main(sys.argv))
