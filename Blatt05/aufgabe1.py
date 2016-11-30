import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import root_pandas as rpd
from sklearn.preprocessing import LabelEncoder
plt.style.use('ggplot')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
plt.rcParams['font.family'] = 'lmodern'

def labelling(exa,label):
	#generates labels for data
	if label==1:
		labels = np.ones(len(exa))
	elif label ==0:
		labels = np.zeros(len(exa))
	return labels


def aufg1():
	###read and label data
	P0 = rpd.read_root('zwei_populationen.root',key='P_0_10000')
	#print(P0)
	P1 = rpd.read_root('zwei_populationen.root',key='P_1')
	
	P0['label'] = labelling(P0,0)
	P1['label'] = labelling(P1,1)
	df = P0.append(P1,ignore_index=True)
	#print(df)
	###compute means
	
	#mean_x_1 = np.mean(P1['x']) 
	#mean_x_0 = np.mean(P0['x'])
	#mean_y_1 = np.mean(P1['y'])
	#mean_y_0 = np.mean(P0['y'])
	X = df[[0,1]].values
	y = df['label'].values
	print(y)
	mean_vectors = []
	mean_vectors = []
	for cl in range(0,2):
		mean_vectors.append(np.mean(X[y==cl], axis=0))
		#print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl]))


	#compute scatter matrix
	S_W = np.zeros((2,2))
	for cl,mv in zip(range(0,1) , mean_vectors):
		class_sc_mat = np.zeros((2,2))            
		for row in  X[y==cl]:
			row , mv = row.reshape(2,1) , mv.reshape(2,1)
			class_sc_mat += (row-mv).dot((row-mv).T)
		S_W += class_sc_mat

	print('within-class Scatter Matrix:\n', S_W)

	overall_mean = np.mean(X, axis=0)

	S_B = np.zeros((2,2))
	for i,mean_vec in enumerate(mean_vectors):  
		n = X[y==i+1,:].shape[0]
		mean_vec = mean_vec.reshape(2,1) # make column vector
		overall_mean = overall_mean.reshape(2,1) # make column vector
		S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

	print('between-class Scatter Matrix:\n', S_B)
	
	eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

	for i in range(len(eig_vals)):
		eigvec_sc = eig_vecs[:,i].reshape(2,1)   
		print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
		print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))

		for i in range(len(eig_vals)):
			eigv = eig_vecs[:,i].reshape(2,1)
			np.testing.assert_array_almost_equal(np.linalg.inv(S_W).dot(S_B).dot(eigv),
				eig_vals[i] * eigv,
				decimal=6, err_msg='', verbose=True)
	print('ok')


	# Make a list of (eigenvalue, eigenvector) tuples
	eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

	# Sort the (eigenvalue, eigenvector) tuples from high to low
	eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

	# Visually confirm that the list is correctly sorted by decreasing eigenvalues

	print('Eigenvalues in decreasing order:\n')
	for i in eig_pairs:
		print(i[0])

	print('Variance explained:\n')
	eigv_sum = sum(eig_vals)
	for i,j in enumerate(eig_pairs):
	    print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))

	W = np.hstack((eig_pairs[0][1].reshape(2,1), eig_pairs[1][1].reshape(2,1)))
	print('Matrix W:\n', W.real)

	X_lda = X.dot(W)
	assert X_lda.shape == (150,2), "The matrix is not 150x2 dimensional."

if __name__ == '__main__':
	aufg1()