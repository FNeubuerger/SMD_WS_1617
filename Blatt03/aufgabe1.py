import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import root_pandas as rpd
plt.style.use('ggplot')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
plt.rcParams['font.family'] = 'lmodern'
from numpy.random import uniform


def aufg1():

	#a)
	def fluxMC(gamma,size):
		return (1-uniform(0,1,size))**(1-gamma)

	fluxMCarr = fluxMC(2.7,1e5)
	#plt.hist(fluxMCarr,bins=np.logspace(1, 10, 40 ),color='navy',rasterized=True)
	#plt.xscale('log')
	#plt.yscale('log')
	#plt.savefig('a_hist.png',dpi=200)
	#plt.show()
	#print(max(fluxMCarr))

	energie = pd.DataFrame({'Energie' : fluxMCarr})
	#rpd.to_root(energie, 'NeutrinoMC.root', key='Signal_MC') #mode='a' for appending

	#b)
	def verwerfungsmethode(rho):
		while True:
			p = fluxMCarr
			u = uniform(0,1)
			if u < rho(p).all():
				return p
	def P_acc(E):
		return (1-np.exp(-E/2))**3
	acceptance = pd.DataFrame({'Energie_Akzeptanz' : verwerfungsmethode(P_acc)})
	#print(acceptance) 
	keys = ['Signal_MC','Signal_MC_Akzeptanz']
	df = pd.concat([energie, acceptance], axis=1)
	print(df)
	df.to_root('NeutrinoMC.root',key='maintree')


if __name__ == '__main__':
	aufg1()
