import numpy as np 
import matplotlib.pyplot as plt 
import root_pandas as rpd
import pandas as pd
from scipy.spatial import distance

plt.style.use('ggplot')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
plt.rcParams['font.family'] = 'lmodern'

def readData(filename,key):
	df = rpd.read_root(filename,key)
	return df

def labelling(exa,label):
	#returns labelled set
	if label==1:
		exa['label'] = np.ones(len(exa))
	elif label ==0:
		exa['label'] = np.zeros(len(exa))
	return exa

def eucdist(train,test):
	'''
	#euclidian distance for all elements in dataframe
	#del set2['label']
	dist = pd.DataFrame()
	for att in test.columns.values.tolist(): #eignetlich wäre hier eine maske für die in beiden df enthaltenen columns gut.
		print(att)
		dist[att] = np.zeros(len(test[att])) #empty df for storage
		for i in range(len(test[att])):
			dist[att][i] += (test[att][i]-train[att][i])**2
	return dist.applymap(lambda x: np.sqrt(x))
	'''
	return train.apply(euclidean_distance, axis=1)


def euclidian_distance(set1,set2):
	inner = 0
	for k in set2.columns.values.tolist():
		inner += (set1[k]-set2[k])**2
	return np.sqrt(inner)

def getNeibours(training,test,k):
	distances = pd.DataFrame()
	length = len(test)-1
	for x in training.columns.values.tolist():
		dist = euclidian_distance(test,training)
		distances[x]=training[x]
		distances['dist'][x]=dist
	distances[].argsort(axis=0)
	k_neighbours = distances.

def k_smallest_ind(df,k):
	#returns the indices of the k entries with the lowest values from fiven df
	indices=pd.DataFrame()
	for att in df.columns.values.tolist():
		indices[att] = df[att].argsort(axis=0)
	return indices.tail(n=k).reset_index(drop=True)  #returns the last k rows since sort is descending order

def k_smallest_labels(df,indices):
	#returns labels of 
	k_smallest=pd.DataFrame()
	result = pd.DataFrame()
	for key in indices.columns.values.tolist():
		k_smallest[key] = df['label'].iloc[indices[key]]
		result = pd.concat([result,k_smallest], axis=1,ignore_index=True).dropna(axis=1)
	#result.columns = indices.columns.values.tolist()
	result=result.reset_index(drop=True) #reset arbitrary indices
	del df['label'] #removes arbitrary label column in result df
	
	return result

def count_values(df):
	#counts all values in df that might appear and returns them as a df with column names
	count=pd.DataFrame()
	for key in df.columns.values.tolist():
		count[key]=df[key].value_counts()
	return count


def knn(signal,bkg,k=10):
	#labelled example sets
	data = signal.append(bkg).reset_index(drop=True)
	#split sets into training and testsets with given samplesize
	
	train_sig = signal.sample(5000)
	train_bg = bkg.sample(5000)

	train = train_sig.append(train_bg).reset_index(drop=True)
	#print(train)
	test_sig=signal.sample(10000)
	test_bg=bkg.sample(20000)

	test = test_sig.append(test_bg).reset_index(drop=True)
	dist1 = train.apply(lambda row: distance.euclidean(row,train))
	#dist1.to_csv('dist.csv')
	#dist1 = pd.read_csv('dist.csv',index_col='Unnamed: 0')
	dist_df = pd.DataFrame(data={'dist':dist1, "idx": euclidean_distances.index})
	dist_df.sort('dist',inplace=True)
	k_nearest = dist_df.head(n=k)

	k_nearest = training.loc[k_nearest.index]["index"]
	print(k_nearest)
	#print('dist: ',dist1)
	#k_ind = k_smallest_ind(dist1,k) 
	#print('k_ind:\n ',k_ind)
	#print(test)
	#smallest_labels=k_smallest_labels(test,k_ind)
	#print(smallest_labels)
	
	#count = count_values(smallest_labels)
	#print(count)


def aufg1():
	sig = readData('NeutrinoMC.root','Signal_MC_Akzeptanz')
	del sig['Energie']
	bg = readData('NeutrinoMC.root','Untergrund_MC')
	sig_l = labelling(sig,1).sample(10000)
	bg_l = labelling(bg,0).sample(20000)
	data = sig_l.append(bg_l).reset_index(drop=True)
	#print(data)
	#train_bg,test_bg=split_set(bg_l,ratio=0.5)



	knn(sig_l,bg_l,k=20)
	'''
	dist1 = eucdist(train,test)
	#dist1.to_csv('dist.csv')
	#dist1 = pd.read_csv('dist.csv',index_col='Unnamed: 0')

	print(dist1)
	k_ind = k_smallest_ind(dist1,20) 
	print(k_ind)
	smallest_labels=k_smallest_labels(data,k_ind)
	print(smallest_labels)
	
	count = count_values(smallest_labels)
	print(count)
	'''
	

if __name__ == '__main__':
	aufg1()
