import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import root_pandas as rpd
plt.style.use('ggplot')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
plt.rcParams['font.family'] = 'lmodern'
from numpy.random import uniform
from matplotlib.colors import LogNorm

def polar():
	q=0
	while q<=0 or q>=1:
		u = 2*uniform() -1
		v = 2*uniform() -1
		q = u**2 + v**2
	p = np.sqrt(-2 * np.log(q)/q )
	x1 = u*p
	x2 = v*p
	return x2 



def aufg1():

	#a)
	def fluxMC(gamma,size):
		return (1-uniform(0,1,size))**(1/(1-gamma))

	fluxMCarr = fluxMC(2.7,int(1e5))
	

	energie = pd.DataFrame({'Energie' : fluxMCarr})
	rpd.to_root(energie, 'NeutrinoMC.root', key='Signal_MC') 
	#b)
	
	u = uniform(0,1,len(fluxMCarr))

	def P_acc(E):
		return (1-np.exp(-E/2))**3

	ACC = P_acc(fluxMCarr)
	acc_arr = fluxMCarr[ACC >= u]
	acceptance = pd.DataFrame({'Energie' : acc_arr})
	#print(acceptance)
	#rpd.to_root(acceptance,'NeutrinoMC.root',key='Signal_MC_Akzeptanz',mode='a')
	
	#c)
	#Normalverteilung mit mu=10E std = 2E
	
	N_hits_arr = np.zeros(len(acc_arr))-1
	for i in range(len(acc_arr)):
		mean = 10*acc_arr[i]
		std = 2*acc_arr[i] 
		while N_hits_arr[i] < 0:
			N_hits_arr[i]=round(std*polar()+mean)#round( np.random.normal(mean,std))
		

	N_hits = pd.DataFrame({'N_hits':N_hits_arr})
	#print(N_hits)
	#rpd.to_root(N_hits,'NeutrinoMC.root',key='Signal_MC_Akzeptanz',mode='a')

	#d)
	
	x1 = np.zeros(len(acc_arr))-1
	x2 = np.zeros(len(acc_arr))-1

	for i in range(len(acc_arr)):
		sigma = 1/np.log10(N_hits_arr[i]+1)
		x1[i] = sigma*polar()+7#np.random.normal(7,sigma)
		x2[i] = sigma*polar()+3#np.random.normal(3,sigma)
		if x1[i]>10 or x2[i]>10 or x1[i]<0 or x2[i]<0:
			i-=1

	position = pd.DataFrame({'x':x1 , 'y':x2})
	#print(position)

	df_acc = pd.concat([acceptance,N_hits,position],ignore_index=True)
	#print(df_acc)
	#rpd.to_root(position,'NeutrinoMC.root',key='Signal_MC_Akzeptanz',mode='a')
	rpd.to_root(df_acc,'NeutrinoMC.root',key='Signal_MC_Akzeptanz',mode='a')
	
	#e)
	#Untergrund MC
	#Anzahl_Hits
	#N_hits_bg_log=[]
	#for i in range(int(1e7)):
	#	N_hits_bg_log.append(np.random.normal(2,1))
	N_hits_bg_log = np.random.normal(2,1,int(1e7))

	N_hits = 10**np.array([N_hits_bg_log]).flatten()
	
	#generate positions
	x1_bg = np.zeros(int(1e7))-1
	x2_bg = np.zeros(int(1e7))-1
	
	#mean = [5,5]
	sigma_x = 3
	sigma_y = 3
	rho = 0.5
	#cov_xy = rho*sigma_x*sigma_y
	#cov = [[sigma_x**2 , cov_xy],[cov_xy, sigma_y**2]]
	#cov matrix not positive semidefinite (runtime warning)
	for i in range(len(N_hits_bg_log)):
		#pos = np.random.multivariate_normal(mean=mean,cov=cov).T
		
		while x1_bg[i]>10 or x2_bg[i]>10 or x1_bg[i]<0 or x2_bg[i]<0:
			y_s = np.random.normal(0,1)
			x_s = np.random.normal(0,1)
			x1_bg[i] = x_s*sigma_x*np.sqrt(1-rho**2) + rho*sigma_y*y_s+5 
			x2_bg[i] = y_s*sigma_y + 5
		#if x1_bg[i]>10 or x2_bg[i]>10 or x1_bg[i]<0 or x2_bg[i]<0:
		#	i-=1


	bg = pd.DataFrame({'N_hits':N_hits,'x':x1_bg ,'y':x2_bg })
	
	rpd.to_root(bg,'NeutrinoMC.root',key='Untergrund_MC',mode='a')



if __name__ == '__main__':
	aufg1()
