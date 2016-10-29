 
import numpy as np 
import matplotlib.pyplot as plt 

plt.style.use('ggplot')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
plt.rcParams['font.family'] = 'lmodern'
def aufg2():
	##########################
	###Definiere Funktionen###
	##########################
	def f(x):
		return (np.sqrt(9-x)-3)/x


	xx = np.logspace(-20,-1,500)
	def c(x): #grenzwert der Funktion
		return -1/6 +0*x
	##########################
	#####Plotte Funktionen####
	##########################

	fig = plt.figure(figsize=(16,9))
	plt.plot(xx,f(xx),'rx',label='f(x)')
	plt.plot(xx,c(xx),'k--',label='analytischer Grenzwert')
	plt.xlabel(r'x')
	plt.ylabel(r'Wert der Funktion')
	plt.xscale('log')
	plt.legend(loc='best')
	plt.savefig("plot2.png",dpi=300)
	plt.show()

if __name__ == '__main__':
	aufg2()