import numpy as np 
import matplotlib.pyplot as plt 

plt.style.use('ggplot')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
plt.rcParams['font.family'] = 'lmodern'
def aufg4():
	a = 7.2973525664e-3
	E_e = 50e9
	m_e = 511e3
	s=(2*E_e)**2
	gamma = E_e/m_e
	b = np.sqrt(1-gamma**-2)
	
	print(b)
	print(gamma)
	def f(x):
		return ((a**2)/s)*(2+(np.sin(x))**2)/(1-(b**2)*(np.cos(x))**2)

	def g(x):
		return ((a**2)/s)*(2+(np.sin(x))**2)/((1/(gamma**2)+b**2*np.sin(x)**2))

	def kond(x):
		return (2*np.cos(x)-2*(b**2)*(2*np.sin(x)-1))*x/(2+np.sin(x)**2)


	xx = np.linspace(np.pi-1e-10,np.pi+1e-10,150)
	yy = np.linspace(0, np.pi, 50)


	fig = plt.figure(figsize=(16,9))
	#plt.plot(xx,f(xx),'ro',markersize=10,label=r'$f(x)$')
	#plt.plot(xx,g(xx),'bx',markersize=10,label=r'$g(x)$')
	plt.plot(yy,abs(kond(yy)),'bx',markersize=10,label=r'$g(x)$')
	
	#plt.xlabel(r'x')
	#plt.ylabel(r'Wert der Funktion')
	#plt.xscale('log')
	#plt.yscale('log')
	#plt.xlim(-1e-2,1e20)
	#plt.ylim(-1e-2 , 0.9)
	plt.legend(loc='best')
	#plt.savefig("plot3.png")
	plt.show()



if __name__ == '__main__':
	aufg4()