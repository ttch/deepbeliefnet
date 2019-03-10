# -*- coding:UTF-8 -*-

import sys
import os
sys.path.append(os.getcwd())

#from dbn import mylogging




# *.* 
import json
import itertools

hidden_layers_structure=[[4000,4000,4000],[2000,2000,2000],[6000,6000,6000]]
learning_rate_rbm=[0.01,0.001,0.05,0.1]
learning_rate=[0.1,0.5,0.2]
n_epochs_rbm=[80,40]
n_iter_backprop=[100]
batch_size=[100,200]
activation_function= ['relu']
dropout_p=[0.1]



hidden_layers_structure_i = range(0,len(hidden_layers_structure))

learning_rate_rbm_i = range(0,len(learning_rate_rbm))

learning_rate_i = range(0,len(learning_rate))

n_epochs_rbm_i = range(0,len(n_epochs_rbm))

n_iter_backprop_i = [0]

batch_size_i = range(0,len(batch_size))

activation_function_i = [0]

dropout_p_i = range(0,len(dropout_p))

l = []

for i in itertools.product(hidden_layers_structure_i, \
						learning_rate_rbm_i,learning_rate_i,n_epochs_rbm_i, \
						n_iter_backprop_i,batch_size_i, \
						activation_function_i,dropout_p_i):
	l.append(i)

co = len(hidden_layers_structure_i) * len(learning_rate_rbm_i) *\
	len(learning_rate_i) * len(n_epochs_rbm_i) * len(n_iter_backprop_i)* \
	len(batch_size_i) * len(activation_function_i) * len(dropout_p_i)

print(len(hidden_layers_structure_i) , len(learning_rate_rbm_i) ,
	len(learning_rate_i) , len(n_epochs_rbm_i) , len(n_iter_backprop_i),
	len(batch_size_i), len(activation_function_i) , len(dropout_p_i))
print(co)
print(len(l))



def writecfg(context):
	f = open("args.cfg","a+")
	f.write(context+"\n")
	f.close()


def main(argv):
	i = 0

	for x in l:
		args = {
			"i":i,
			"hidden_layers_structure" : hidden_layers_structure[x[0]],
			"learning_rate_rbm" :  learning_rate_rbm[x[1]],
			"learning_rate" : learning_rate[x[2]],
			"n_epochs_rbm" : n_epochs_rbm[x[3]],
			"n_iter_backprop" : n_iter_backprop[x[4]],
			"batch_size" : batch_size[x[5]],
			"activation_function" : activation_function[x[6]],
			"dropout_p" : dropout_p[x[7]]
		}

		
		writecfg(json.dumps(args))


		i = i +1




if __name__ == '__main__':
	main(sys.argv)
