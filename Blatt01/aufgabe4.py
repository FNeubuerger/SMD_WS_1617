import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.patches import Ellipse
import matplotlib as matplotlib
import numpy.random as rnd
plt.style.use('ggplot')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
plt.rcParams['font.family'] = 'lmodern'
def aufg4():

	rho = 0.8
	sigma_1 = 3.5
	sigma_2 = 1.5
	k = 4.2
	alpha = 1/2*np.arctan(2*rho*sigma_1*sigma_2/(sigma_1**2-sigma_2**2))
	mean = [4, 2]
	cov = [[sigma_1**2, k], [sigma_2**2, k]] 
	fig = plt.figure(0)
	x, y = np.random.multivariate_normal(mean, cov, 700).T
	a_1=4
	a_2=2
	u_1 = (x-a_1)/sigma_1
	u_2 = (x-a_2)/sigma_2
	print(a_1,a_2)
	
	p_1 = (1-rho**2)*2*(np.cos(alpha)**2/sigma_1**2-2*rho*np.sin(alpha)*np.cos(alpha)/sigma_1*sigma_2+np.sin(alpha)**2/sigma_2**2)**(-1)

	p_2 = (1-rho**2)*2*(np.sin(alpha)**2/sigma_1**2-2*rho*np.sin(alpha)*np.cos(alpha)/sigma_1*sigma_2+np.cos(alpha)**2/sigma_2**2)**(-1)
	
	print(p_1,p_2)

	#ells = [matplotlib.patches.Ellipse((a_1*a_2), width=p_1, height=p_2, angle=alpha, color='k')]
	plt.plot(x, y, 'x')
	
	#plt.show()
	plt.axes()
	#plt.figure()
	ax = plt.gca()

	ellipse = Ellipse(xy=(a_1, a_2), width=p_1, height=p_2,angle=np.rad2deg(alpha),edgecolor='k', fc='None')#, lw=2)
	ellipse_2 = Ellipse(xy=(a_1, a_2), width=p_1*1/np.sqrt(np.e), height=p_2*1/np.sqrt(np.e),angle=np.rad2deg(alpha),edgecolor='g', fc='None')#, lw=2)
	plt.gca().add_patch(ellipse)
	plt.gca().add_patch(ellipse_2)
	errX = sigma_1+a_1
	errY = sigma_2+a_2
	plt.errorbar(a_1 , a_2 , xerr=errY, yerr=errY, fmt='x')
	plt.xlim(-16,16)
	plt.ylim(-11,11)
	plt.show()






if __name__ == '__main__':
	aufg4()



