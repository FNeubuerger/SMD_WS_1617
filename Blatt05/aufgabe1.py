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

def predict(projected_x,n_cuts,cut_min,cut_max):
	#binary prediction by different cuts
	cuts = np.linspace(cut_min,cut_max,n_cuts)
	pred_cuts=[]
	for i in range(len(cuts)):
		pred = [] #muss der cut auch auf die gerade projiziert werden?
		for x in projected_x: 
			if x < cuts[i]:
				pred.append(0)
			else:
				pred.append(1)
		pred_cuts.append(pred)
	return pred_cuts,cuts #returns list of predictionlists for each cut

def performance(prediction,label):
	#calculates precision, recall and accuracy of given prediction and label
	prediction = np.array(prediction)
	
	label = np.array(label)
	right = prediction[prediction==label]
	wrong = prediction[prediction!=label]
	sig = len(prediction[prediction==1]) #number of signal
	
	bkg = len(prediction[prediction==0])
	#if bkg !=0:
		#print(sig/bkg)
	ratio=[]
	significance=[]
	if bkg!=0:
		ratio.append((sig/bkg))
	else:
		ratio.append(0)
	#print(ratio[0])
	significance.append(sig/np.sqrt(sig+bkg))
	#print(significance[0])
	#number of tp fp tn and fn
	tp = len(right[right==1])
	fp = len(wrong[wrong==1])
	tn = len(right[right==0])
	fn = len(wrong[wrong==0])
	
	
	if tp==0:
		precision = 0
		if fp==0: 
			precision=1
	else:
		precision =  tp/(tp+fp)
	if tp==0 and fn==0:
		recall = 0
	else:
		recall = tp/(tp+fn)
	if tp+tn==0:
		accuracy=0
	else:			
		accuracy = (tp+tn)/(tp+tn+fn+fp)		
	return precision,recall,accuracy,ratio[0],significance[0]


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
	
	X = df[[0,1]].values
	y = df['label'].values
	mean_vectors = []

	for cl in range(0,2):
		mean_vectors.append(np.mean(X[y==cl], axis=0))
		print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl]))


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


	# Make a list of (eigenvalue, eigenvector) tuples
	eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

	# Sort the (eigenvalue, eigenvector) tuples from high to low
	eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
	#transformationsmatrix
	W = np.hstack((eig_pairs[0][1].reshape(2,1), eig_pairs[1][1].reshape(2,1)))
	print('Matrix W:\n', W.real)

	X_lda = X.dot(W) #x werte der klassen projiziert auf geraden
	#plot 1d hists for every class
	
	label_dict = {0: 'P1', 1: 'P1'}
	for lin,marker,color in zip(
		range(0,2),('x', 'o'),('blue', 'red')):
		plt.hist(X_lda[:,lin].real[y == 0],alpha=0.5,color='red',label='P0',bins=np.linspace(-6,10,17))
		plt.hist(X_lda[:,lin].real[y == 1],alpha=0.5,color='navy',label='P1',bins=np.linspace(-6,10,17))
		plt.legend(loc='upper right',fancybox=True)
		plt.savefig('hist_'+str(lin)+'.png')
		#plt.show()
		plt.clf()

	ncuts = 100
	pred_g1,cuts_g1 = predict(X_lda[:,0],n_cuts=ncuts,cut_min=-10,cut_max=10)
	
	prec_g1=[]
	rec_g1=[]
	acc_g1=[]
	ratio=[]
	significance=[]
	for i in range(len(pred_g1)):

		prec1 , rec1 , acc1, ratio1, significance1 = performance(pred_g1[i],df['label'].values)
		prec_g1.append(prec1)
		rec_g1.append(rec1)
		acc_g1.append(acc1)
		ratio.append(ratio1)
		significance.append(significance1)

	pred_g2,cuts_g2 = predict(X_lda[:,1],n_cuts=ncuts,cut_min=-10,cut_max=10)
	
	prec_g2=[]
	rec_g2=[]
	acc_g2=[]
	ratio2=[]
	significance2=[]
	for i in range(len(pred_g2)):

		prec1 , rec1 , acc1, ratio1, significance1 = performance(pred_g2[i],df['label'].values)
		prec_g2.append(prec1)
		rec_g2.append(rec1)
		acc_g2.append(acc1)
		ratio2.append(ratio1)
		significance2.append(significance1)
	
	l_max_ratio = cuts_g1[ratio==np.max(ratio)]
	print('lambda_max ratio:\n',l_max_ratio)
	l_max_sig = cuts_g1[significance==np.max(significance)]
	print('lambda_max significance:\n',l_max_sig)

	l_max_ratio2 = cuts_g2[ratio2==np.max(ratio2)]
	print('lambda_max ratio:\n',l_max_ratio2)
	l_max_sig2 = cuts_g2[significance2==np.max(significance2)]
	print('lambda_max significance:\n',l_max_sig2)

	plt.clf()
	plt.plot(cuts_g1,ratio)
	plt.xlabel(r'$x_{\text{cut}}$')
	plt.ylabel(r'$S/B$')
	plt.yscale('log')
	plt.savefig('sig_bkg_ratio.png')
	#plt.show()
	plt.clf()
	plt.plot(cuts_g1,significance)
	plt.xlabel(r'$x_{\text{cut}}$')
	plt.ylabel(r'Signifikanz')
	plt.yscale('log')
	plt.savefig('signifikanz.png')
	#plt.show()

	plt.clf()
	plt.plot(cuts_g2,ratio2)
	plt.xlabel(r'$x_{\text{cut}}$')
	plt.ylabel(r'$S/B$')
	plt.yscale('log')
	plt.savefig('sig_bkg_ratio2.png')
	#plt.show()
	plt.clf()
	plt.plot(cuts_g2,significance2)
	plt.xlabel(r'$x_{\text{cut}}$')
	plt.ylabel(r'Signifikanz')
	plt.yscale('log')
	plt.savefig('signifikanz2.png')
	#plt.show()


	plt.clf()
	plt.plot(cuts_g1,prec_g1,label='precision',rasterized=True)
	plt.plot(cuts_g1,rec_g1,label='recall',rasterized=True)
	plt.plot(cuts_g1,acc_g1,label='accuracy',rasterized=True)
	plt.legend(loc='best',fancybox=True)
	plt.ylim(-0.01,1.01)
	plt.savefig('performace_1.png',dpi=100)
	#plt.show()
	plt.clf()
	plt.plot(cuts_g2,prec_g2,label='precision',rasterized=True)
	plt.plot(cuts_g2,rec_g2,label='recall',rasterized=True)
	plt.plot(cuts_g2,acc_g2,label='accuracy',rasterized=True)
	plt.legend(loc='best',fancybox=True)
	plt.ylim(-0.01,1.01)
	plt.savefig('performace_2.png',dpi=100)
	#plt.show()

	

if __name__ == '__main__':
	aufg1()