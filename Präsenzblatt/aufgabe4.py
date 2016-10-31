import ROOT as root
import root_numpy as rnp
import numpy as np 	
def aufg4():
	#root_file = root.TFile("TFile.root", "RECREATE")

	x = np.random.uniform(1,1000,size=1000)
	y = np.random.normal(x,np.sqrt(x))
	z = np.random.poisson(x)

	arr = np.empty((len(x),),dtype=[('x',float),('y',float),('z',int)])
	arr['x']=x
	arr['y']=y
	arr['z']=z
	#print(arr)
	rnp.array2root(arr,'test.root',treename='master',mode='RECREATE')
	#rnp.array2root(x,'TFile.root','x','RECREATE')
	#rnp.array2root(y,'TFile.root','y','RECREATE')
	#rnp.array2root(z,'TFile.root','z','RECREATE')

if __name__ == '__main__':
	aufg4()