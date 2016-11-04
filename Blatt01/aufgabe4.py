import numpy as np 
import matplotlib.pyplot as plt 

plt.style.use('ggplot')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
plt.rcParams['font.family'] = 'lmodern'
def aufg4():

	
	#############################
	#####Definiere Konstanten####
	#############################


	a = 1/137
	E_e = 50e9
	m_e = 511e3
	s=(2*E_e)**2
	gamma = E_e/m_e
	b = np.sqrt(1-gamma**-2)
	
	#############################
	#####Definiere Funktionen####
	#############################


	#usrpüngliche Funktion
	def f(x):
		return ((a**2)/s)*(2+(np.sin(x))**2)/(1-(b**2)*(np.cos(x))**2)
	
	#Funktion nach Umformung
	def g(x):
		return ((a**2)/s)*(2+(np.sin(x))**2)/((1/(gamma**2)+b**2*np.sin(x)**2))

	#Gleichung Konditionszahl der Funktion nach der Umformung
	def kond(x):
		return ((4*a**2*gamma**2*(-2*gamma**2+3)*np.sin(2*x))/(s*(-gamma**2*np.cos(2*x)+gamma**2+np.cos(2*x)+1)**2))*x
	

	##########################
	#####Plotte Funktionen####
	##########################


	xx = np.pi*(np.arange(1000,dtype=float) + 1)*2*10**(-11)
	yy = np.linspace(0, np.pi, 50)
	zz = np.arange(0, np.pi, 0.000001)


	fig = plt.figure(figsize=(12,8))
	plt.plot(xx,f(xx),'r.',markersize=10,label=r'$\frac{d\sigma}{d\Omega}$', alpha=0.5)
	plt.xlabel(r'$\theta$')
	plt.ylabel(r'$\frac{d\sigma}{d\Omega}$')
	plt.rcParams.update({'font.size': 22})
	plt.legend(loc='best')
	plt.tight_layout()
	plt.savefig("plot4_c1.png")
	#plt.show()


	fig = plt.figure(figsize=(12,8))
	plt.plot(xx,g(xx),'b.',markersize=10,label=r'$\frac{d\sigma}{d\Omega}_{neu}$' ,alpha=0.5)	
	plt.xlabel(r'$\theta$')
	plt.ylabel(r'diff. Wirkungsquerschnitt')
	plt.legend(loc='best')
	plt.tight_layout()
	plt.savefig("plot4_c2.png")
	#plt.show()
	


	fig = plt.figure(figsize=(12,8))
	plt.plot(zz,abs(kond(zz)),'bx',markersize=10,label=r'Konditionszahl')
	plt.xlabel(r'$\theta$')
	plt.ylabel(r'Konditionszahl')
	plt.xlim(0, 1.1*np.pi)
	plt.legend(loc='best')
	plt.tight_layout()
	plt.savefig("plot4_e.png")
	#plt.show()
	plt.clf()


	fig = plt.figure(figsize=(12,8))
	plt.plot(zz,abs(kond(zz)),'bx',markersize=10,label=r'Konditionszahl')
	plt.xlabel(r'$\theta$')
	plt.ylabel(r'Konditionszahl')
	plt.xlim(0, 1.1*np.pi)
	plt.yscale('log')
	plt.legend(loc='best')
	plt.tight_layout()
	plt.savefig("plot4_elog.png")
	#plt.show()
	plt.clf()



if __name__ == '__main__':
	aufg4()
