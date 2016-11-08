
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import scipy.stats as scs
def aufg4():

	##########################
	###Definiere Konstanten###
	##########################

	rho = 0.8 
	sigma_1 = 3.5
	sigma_2 = 1.5
	k = 4.2 	#cov(x,y)
	a_1=4		#mu_1 vom Zettel
	a_2=2		#mu_2 vom Zettel 
	
	mean = [a_1, a_2]	
	cov = [[sigma_1**2, k], [sigma_2**2, k]] 
	
	#Winkel zwischen Ellipse und Koordinatensystem in Radiant
	alpha = 1/2*np.arctan(2*rho*sigma_1*sigma_2/(sigma_1**2-sigma_2**2))
	
	#Halbachsen**2

	p_1 = 2*np.sqrt((1-rho**2))*(np.sin(alpha)**2/sigma_1**2-2*rho*np.sin(alpha)*np.cos(alpha)/sigma_1*sigma_2+np.cos(alpha)**2/sigma_2**2)**(-1/2)
	p_2 = 2*np.sqrt((1-rho**2))*(np.sin(alpha)**2/sigma_1**2+2*rho*np.sin(alpha)*np.cos(alpha)/sigma_1*sigma_2+np.cos(alpha)**2/sigma_2**2)**(-1/2)	
	

	x = np.linspace(-15,15, 1000)
	y = np.linspace(-15, 15, 1000)
	u_1 = (x-a_1)/sigma_1
	u_2 = (x-a_2)/sigma_2

	##############
	###2d Gauss###
	##############
	xx, yy = np.meshgrid(x, y)
	# Get the 2d GauÃ values
	XX = xx.flatten()
	YY = yy.flatten()

	# Scipy needs the x, y values as: [[x1, y1], [x2, y2], ..., [xN, yN]]
	vals = list(zip(XX, YY))
	gaus_pdf = scs.multivariate_normal.pdf(
	vals, mean=[4, 2], cov=[[sigma_1**2, k], [sigma_2**2, k]])

	# Now reshape for pcolormesh
	zz = gaus_pdf.reshape(xx.shape)
	fig = plt.figure(0)
	# And plot it
	plt.xlim(-2,10)
	plt.ylim(-2,7)
	plt.pcolormesh(xx, yy, zz, cmap="hot", label='$p_1,p_2$')
	

	#Steigung der Geraden für Einzeichnen der Halbachsen 
	m_1=np.tan(2*np.pi*(np.rad2deg(alpha))/360)
	m_2=np.tan(2*np.pi*(np.rad2deg(alpha)+90)/360)
	
	#Plotten der Geraden u und v 
	u = np.linspace(2.69,4)
	plt.plot(u, 2-m_1*4+m_1*u, 'k--', label='Halbachsen')
	v = np.linspace(3.75, 4) 
	plt.plot(v, 2-m_2*4+m_2*v, 'k--')
	#print(p_1/2,p_2/2)
	#print(np.rad2deg(alpha))
	#Ellipsen plotten 
	plt.axes()
	ax = plt.gca()

	#1 sigma Ellipse
	ellipse = Ellipse(xy=(a_1, a_2), width=p_1, height=p_2,angle=np.rad2deg(alpha),edgecolor='k', fc='None') # Ellipse 
	#Ellipse mit p_1, p_2 auf 1/sqrt(e) abgesunken
	ellipse_2 = Ellipse(xy=(a_1, a_2), width=p_1*1/np.sqrt(np.e), height=p_2*1/np.sqrt(np.e),angle=np.rad2deg(alpha),edgecolor='g', fc='None') 

	plt.gca().add_patch(ellipse)
	plt.gca().add_patch(ellipse_2)
	#Errorbars dazu
	errX = sigma_1
	errY = sigma_2
	plt.errorbar(a_1 , a_2 , xerr=errY, yerr=errY, fmt='x',label="$\mu_x\pm\sigma_x,\mu_y\pm\sigma_y$")
	plt.xlabel(r'x')
	plt.ylabel(r'y')
	plt.plot(0,0,'k-',label='$1\sigma$Ellipse')
	plt.plot(0,0,'g-',label='Abfall auf 1/e$^{1/2}$')
	plt.legend(loc='best')
	plt.colorbar().set_label('$p_1,p_2$' )
	plt.savefig("plot_4e.png", dpi=150)
	plt.show()


	
	#plt.plot(x, y, 'x') # 2d Gauß
	






if __name__ == '__main__':
	aufg4()



