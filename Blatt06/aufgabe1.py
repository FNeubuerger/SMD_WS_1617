import numpy as np 
import matplotlib.pyplot as plt
import root_pandas as rpd
import pandas as pd
from ggplot import ggplot, aes, geom_point

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

def aufg1():
	
	#print(data)
	#train_bg,test_bg=split_set(bg_l,ratio=0.5)
	sig ,bg = read_data()
	test,train = split_sets(sig,bg,100,100,100,200)
	data=test.append(train)
	test_labels=test['label']
	train_labels=train['label']
	#print(data)
	#plot(data,'x','y',color='label')
	result = test.apply(lambda row: knn(row,train,train_labels,k=20),axis=1)
	#result.columns=['0','1']
	#print(result)
	

	errors = result==test_labels 
	
	#print(errors) #tp wo true UND label=1...
	print(errors.value_counts())

	err_labels = pd.concat([errors,test_labels],axis=1)
	err_labels.columns=['prediction','label']
	print(err_labels)
	'''
	if err_labels['prediction']==True and err_labels['label']==1
		tp.append(1) #usw...
	'''
	

if __name__ == '__main__':
	aufg1()
