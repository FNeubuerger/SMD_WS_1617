 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import cauchy
from scipy import stats  
plt.style.use('ggplot')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
plt.rcParams['font.family'] = 'lmodern'


from numpy.random import uniform

def a(xmin,xmax,size):
	return (xmax-xmin)*uniform(0,1,size) + xmin

def b(tau,size):
	return -tau*np.log(1-uniform(0,1,size))

def c(xmin,xmax,n,size):
	if n <2 :
		print('n must be larger or equal to than 2')
		return 0
	return ( uniform(0,1,size) * (xmax**(1-n) - xmin**(1-n)) +xmin**(1-n))**(n-1)

def d(xmin,xmax,size):
	return np.tan(uniform(0,1,size) - np.pi/2)

def aufg3():
	plt.hist(c(xmin=1,xmax=1000,n=5,size=10000),bins=25)
	plt.show()



if __name__ == '__main__':
	aufg3()
