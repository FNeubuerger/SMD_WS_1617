import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import axes3d
import ROOT as R
plt.style.use('ggplot')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
plt.rcParams['font.family'] = 'lmodern'
#a)
def rd_lk(a,b,m,seed=0):
	x_a = seed #whatever seed value default 0
	arr = [] #empty list for storage

	for i in range(m): #loop for generation
		x_n =(a*x_a+b)%m #generates linear kongruent random numbers
		arr.append(x_n/m) #append normed values
		x_a = x_n #set new seed

	return arr #returns array of normed random numbers 

def pair(L):
	return [(L[i],L[i+1])  for i in range(len(L)-1)] #[for j in range(steps-1)]

def triple(L):
	return [(L[i],L[i+1],L[i+2])  for i in range(len(L)-2)]

def ntupel(L,n):
	return [tuple([L[i+j] for j in range(n)]) for i in range(len(L)-n)] #maximale abstraktion



def aufg2():
	#b)
	a = rd_lk(a=1601,b=3456,m=10000)
	#b = rd_lk(a=1601,b=3456,m=10000,seed=50) #unterscheidet sich nur wenn seed nicht in voherigem array enthalten
	#print(a)
	fig = plt.figure(figsize=(16,9))
	plt.hist(a,bins=100,rasterized=True,color='red' ,alpha=0.5)
	#plt.hist(b,bins=100,rasterized=True,color='navy',alpha=0.5)
	plt.xlabel(r'random number value')
	plt.ylabel(r'n')
	#plt.savefig('rd_lk_2.png',dpi=300)
	#plt.show()
	plt.clf()
	test = [0,4,2,6,4,8,6,10,8,9,10,11,12,13,14]
	#c)
	pairs = ntupel(a,2) #pair(a)
	triplel = ntupel(a,3) # triple(a)
	
	#print(pairs)
	#print(triplel)
	

	fig = plt.figure()
	ax1 = fig.add_subplot(111,projection='3d')
	#x , y , z = np.random.normal(size=(3,1000))
	x , y , z = zip(*triplel)
	
	
	ax1.scatter(x,y,z, lw=0)
	#ax1.init_view(45, 30)
	plt.savefig('3dscatter.png',dpi=300)
	#plt.show()
	plt.clf()

	fig = plt.figure(figsize=(16,9))
	X,Y = zip(*pairs)
	plt.scatter(X,Y)
	plt.savefig('2dscatter.png',dpi=300)
	#plt.show()
	plt.clf()

	#d)

	root_random = np.zeros(10000)
	myGen = R.TRandom()
	myGen.RndmArray(10000,root_random)
	print(root_random)

	#e)
	root_pairs = ntupel(root_random,2)

	root_triple = ntupel(root_random,3)

	fig = plt.figure()
	ax1 = fig.add_subplot(111,projection='3d')
	#x , y , z = np.random.normal(size=(3,1000))
	x , y , z = zip(*root_triple)
	
	
	ax1.scatter(x,y,z, lw=0)
	#ax1.init_view(45, 30)
	plt.savefig('3dscatter_root.png',dpi=300)
	#plt.show()
	plt.clf()

	fig = plt.figure(figsize=(16,9))
	X,Y = zip(*root_pairs)
	plt.scatter(X,Y)
	plt.savefig('2dscatter_root.png',dpi=300)
	#plt.show()
	plt.clf()
	#f) nur wenn seed/m = 0.5 


if __name__ == '__main__':
	aufg2()
