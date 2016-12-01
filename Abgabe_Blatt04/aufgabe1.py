import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import root_pandas as rpd
plt.style.use('ggplot')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
plt.rcParams['font.family'] = 'lmodern'



def aufg1():
	mean_0 = [0,3]
	rho_0 = 0.9
	sigma_y = 2.6
	sigma_x = 3.5
	cov_0_xy = rho_0*sigma_y*sigma_x
	
	cov_0 = [[sigma_x**2,cov_0_xy],[cov_0_xy,sigma_y**2]]

	P0 = np.random.multivariate_normal(mean=mean_0, cov=cov_0,size=10000).T


	xx = np.random.normal(6,3.5,10000)

	a = -0.5
	b = 0.6
	mean_1 = a + b*xx
	sigma_1 = 1

	y1 = np.random.normal(mean_1, sigma_1)
	#XX,YY = np.meshgrid(xx,y1)
	#print(P0)

	plt.scatter(P0[0],P0[1],color='navy',s=1,alpha=0.5)
	plt.scatter(xx,y1,color='red',s=1,alpha=0.5)
	plt.xlabel(r'x')
	plt.ylabel(r'y')
	plt.savefig('1b.png',dpi=200)
	#plt.show()

	P0_2 = np.random.multivariate_normal(mean=mean_0, cov=cov_0,size=1000).T

	P0 = pd.DataFrame({'x':P0[0],'y':P0[1]})
	P1 = pd.DataFrame({'x':xx , 'y':y1})
	P0_2 = pd.DataFrame({'x':P0_2[0],'y':P0_2[1]})

	#df = pd.DataFrame({'P0_10000':P0 , 'P1':P1, 'P0_1000':P0_2})
	rpd.to_root(P0,'zwei_populationen_1.root',key='P0_10000')
	rpd.to_root(P1,'zwei_populationen_1.root',key='P1',mode='a')
	rpd.to_root(P0_2,'zwei_populationen_1.root',key='P0_1000',mode='a')
	
	x_mean = np.mean(xx)
	print(x_mean)
	y_mean = np.mean(y1)
	print(y_mean)
	x_var = np.std(xx)**2
	print(x_var)
	y_var = np.std(y1)**2
	print(y_var)
	XY=np.vstack((xx,y1))
	covXY = np.cov(XY)
	print(covXY)
	corrcoeff = np.corrcoef(xx,y1)
	print(corrcoeff)



if __name__ == '__main__':
	aufg1()