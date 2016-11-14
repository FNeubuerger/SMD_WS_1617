 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import cauchy
from scipy import stats  
plt.style.use('ggplot')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
plt.rcParams['font.family'] = 'lmodern'


from numpy.random import uniform
def verwerfungsmethode(**kwargs):
	while True:
		size = kwargs.pop('n')
		p = uniform(0, 1, size)
		u = uniform(0, 1)
		if u < kwargs.pop('func')(**kwargs):
			return p

def uniform_min_max(**kwargs):
	return uniform(low=kwargs.pop('xmin'),high=kwargs.pop('xmax'),size=kwargs.pop('n'))

def minmax(**kwargs):
	xmin = kwargs.pop('xmin')
	xmax = kwargs.pop('xmax')
	x = np.linspace(xmin,xmax,(xmax-xmin+1))
	return 1+0*x

def exponential(**kwargs):
	if min(t)<0:
		print('Array t contains values smaller than zero. Must be larger or equal to zero')
		return 0
	else:
		return kwargs.pop('N_pot')*np.exp(-kwargs.pop('t')/kwargs.pop('tau'))

def pot(**kwargs):
	xmin = kwargs.pop('xmin')
	xmax = kwargs.pop('xmax')
	exponent = kwargs.pop('exponent')
	N = kwargs.pop('N_pot')
	x = np.linspace(x_min,xmax,(xmax-xmin+1)*steps)
	return N*x**(-exponent)

def cauchy(**kwargs):
	x = np.logspace(-20,20,kwargs.pop('n')+1)
	return 1/np.pi * 1/(1+x**2)

def a(xmin,xmax,size):
	while True:
		p = uniform(0, 1, size)
		u = uniform(0, 1)
		def minmax(p):
			x = np.linspace(xmin,xmax,size)
			return 1+0*x
		if u < minmax(p):
			return p

def b(N,tau,size):
	while True:
		p = uniform(0, 1, size)
		u = uniform(0, 1)
		def minmax(p):
			x = np.linspace(0,1e20,1e20+1)
			return N * np.exp(-x/tau)
		if u < exp(p):
			return p

def aufg3():
	xmin = 0
	xmax = 5
	arr_a = a(xmin=xmin,xmax=xmax,n=1000)
	print(a)


if __name__ == '__main__':
	aufg3()
