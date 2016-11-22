import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import root_pandas as rpd
from matplotlib.colors import LogNorm
#plt.style.use('ggplot')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
plt.rcParams['font.family'] = 'lmodern'
def aufg1():
	
	df = rpd.read_root('NeutrinoMC.root', 'Untergrund_MC')
	#print(df)
	
	
	N_hits_bg_log = np.log10(df['N_hits'])
	plt.hist(N_hits_bg_log,bins=50,rasterized=True)
	plt.xlabel(r'$\log(N_{hits})$')
	plt.ylabel(r'Anzahl')
	#plt.yscale('log')
	plt.savefig('N_hits_log.png',dpi=200)

	#plt.show()
	plt.clf()
	
	x1_bg = df['x']

	x2_bg = df['y']
	#print(x1_bg[x1_bg==x2_bg])
	plt.hist2d(x1_bg, x2_bg,bins=50, norm=LogNorm(),cmap='viridis',rasterized=True)
	plt.colorbar()
	plt.xlabel(r'$x_1$')
	plt.ylabel(r'$x_2$')
	plt.xlim(0,10)
	plt.ylim(0,10)
	plt.savefig('x1_x2_hist.png',dpi=200)
	#plt.show()
	plt.clf()
	
	sig = rpd.read_root('NeutrinoMC.root', 'Signal_MC_Akzeptanz')
	#print(sig)
	x1=sig['x'].dropna()
	x2=sig['y'].dropna()
	#print(x1)
	plt.hist2d(x1, x2,bins=50,norm=LogNorm(),cmap='viridis',rasterized=True)
	plt.colorbar()
	plt.xlabel(r'$x_1$')
	plt.ylabel(r'$x_2$')
	plt.xlim(0,10)
	plt.ylim(0,10)
	plt.savefig('x1_x2_hist_signal.png',dpi=200)
	#plt.show()
	plt.clf()

	fluxMC = rpd.read_root('NeutrinoMC.root', 'Signal_MC')
	fluxMCarr = fluxMC['Energie'].dropna()
	acc = rpd.read_root('NeutrinoMC.root', 'Signal_MC_Akzeptanz')
	acc_arr=acc['Energie'].dropna()

	plt.hist(fluxMCarr,bins=np.logspace(0, 3, 50),color='navy',alpha=0.5,rasterized=True,histtype='stepfilled')
	plt.hist(acc_arr,bins=np.logspace(0, 3, 50),color='red',alpha=0.5,rasterized=True,histtype='stepfilled')
	#plt.ylim(1e-1,1e5)

	plt.xlabel(r'$Energie/TeV$')
	plt.ylabel(r'$Anzahl$')
	plt.xscale('log')
	plt.yscale('log')
	plt.savefig('b_hist.png',dpi=200)
	
if __name__ == '__main__':
	aufg1()