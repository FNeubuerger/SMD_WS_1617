 
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
	return

def d(xmin,xmax,size):
	return np.tan(a(xmin,xmax,size) - np.pi/2)

def aufg3():
	plt.hist(d(xmin=-1,xmax=1,size=100),bins=25)
	plt.show()



if __name__ == '__main__':
	aufg3()
