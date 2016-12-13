import numpy as np 
import matplotlib.pyplot as plt
import root_pandas as rpd
import pandas as pd
from ggplot import ggplot, aes, geom_point
import time
start_time = time.time()
plt.style.use('ggplot')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
plt.rcParams['font.family'] = 'lmodern'


def readroot(filename,key):
	df = rpd.read_root(filename,key)
	return df

def labelling(exa,label):
	#returns labelled set
	if label==1:
		exa['label'] = np.ones(len(exa))
	elif label ==0:
		exa['label'] = np.zeros(len(exa))
	return exa

def read_data():
	sig = readroot('NeutrinoMC.root','Signal_MC_Akzeptanz')
	del sig['Energie'] #energie spalte fehlt in bg
	bg = readroot('NeutrinoMC.root','Untergrund_MC')
	sig_l = labelling(sig,1)
	bg_l = labelling(bg,0)
	data = sig_l.append(bg_l).reset_index(drop=True)
	return sig_l , bg_l

def split_sets(signal,bkg,train_sig=5000,train_bg=5000,test_sig=10000,test_bg=20000):
	#labelled example sets
	data = signal.append(bkg).reset_index(drop=True)
	#split sets into training and testsets with given samplesize
	
	train_sig = signal.sample(train_sig)
	train_bkg = bkg.sample(train_bg)

	train = train_sig.append(train_bkg).reset_index(drop=True)
	#print(train)
	test_sig=signal.sample(test_sig)
	test_bkg=bkg.sample(test_bg)

	test = test_sig.append(test_bkg).reset_index(drop=True)
	return train,test

def normalize(df):
    #normalize data
    min_values = df.min()
    max_values = df.max()
    range_values = max_values - min_values
    norm_df = (df - min_values) / range_values
    return norm_df, range_values, min_values
def knn(test_data,training_data,labels,k=10):
	
	dist = training_data-test_data
	squared_dist = dist**2
	euclidian_dist = squared_dist.sum(axis=1)**0.5
	dist_df = pd.concat([euclidian_dist,labels],axis=1)
	dist_df.columns = ['distance','label']
	#print(dist_df)
	dist_sorted=dist_df.sort_values(by='distance',ascending=True) #geht das auch mit argsort?
	k_nearest = dist_sorted.head(n=k)
	#print(k_nearest)
	return k_nearest['label'].value_counts().index.values[0] #gibt anzahl der label der k nächsten nachbarn aus

def plot(df, x, y, color):
    """
    Scatter plot with two of the features (x, y) grouped by classification (color)
    Args:
        df: Dataframe of data
        x: Feature to plot on x axis
        y: Feature to plot on y axis
        color: Group by this column
    """
    print(ggplot(df, aes(x=x, y=y, color=color)) + geom_point())

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
	
	#print(data)
	#train_bg,test_bg=split_set(bg_l,ratio=0.5)
	sig ,bg = read_data()
	test,train = split_sets(sig,bg)
	data=test.append(train)
	test_labels=test['label']
	train_labels=train['label']
	#print(data)
	#plot(data,'x','y',color='label')
	#hier könnte man noch mal über k iterieren
	result = test.apply(lambda row: knn(row,train,train_labels,k=10),axis=1)
	#result.columns=['0','1']
	#print(result)
	

	#errors = result==test_labels 
	
	#print(errors) #tp wo true UND label=1...
	#print(errors.value_counts())

	#err_labels = pd.concat([result,test_labels],axis=1)
	#err_labels.columns=['prediction','label']
	#print(err_labels)
	'''
	if err_labels['prediction'][i]==True and err_labels['label'][i]==1
		tp.append(1) #usw...
	oder if errors[i]==True ...

	oder bool durch int ersetzen oder andersrum und maske
	
	tp = []
	fp = []
	tn = []
	fn = []

	for i in range(len(err_labels)):
		if err_labels['prediction'][i]==True and err_labels['label'][i]==1 :
			tp.append(1)

		if err_labels['prediction'][i]==True and err_labels['label'][i]==0 :
			fp.append(1)

		if err_labels['prediction'][i]==False and err_labels['label'][i]==0 :
			tn.append(1)

		if err_labels['prediction'][i]==False and err_labels['label'][i]==1 :
			fn.append(1)
	


	print('tp: ', tp)
	print('fp: ', fp)
	print('tn: ', tn)
	print('fn: ', fn)
	'''
	prec, rec, acc, ratio, significance = performance(result,test_labels)
	print('Reinheit ',prec)
	print('Effizienz',rec)
	print('Accuracy ',acc)
	print('Signifikanz: ',significance)

	'''
	Reinheit  0.971604447974583
	Effizienz 0.9786
	Accuracy  0.975
	Signifikanz:  50.36
	---114.47622203826904 seconds---
	'''

	print('---%s seconds---' %(time.time() -start_time))
if __name__ == '__main__':
	aufg1()
