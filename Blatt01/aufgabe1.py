import numpy as np 
import matplotlib.pyplot as plt 

plt.style.use('ggplot')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
plt.rcParams['font.family'] = 'lmodern'
def aufg1():
	##########################
	###Definiere Funktionen###
	##########################
	def f(x):
		return (1-x)**6

	def horner(x):
		return x *(x *(x *(x*((x-6)*x+15)-20)+15)-6)+1

	def expanded(x):
		return x**6-6*x**5+15*x**4-20*x**3+15*x**2-6*x+1

	xx = np.linspace(0.999,1.001,100)

	##########################
	#####Plotte Funktionen####
	##########################

	fig = plt.figure(figsize=(16,9))
	ax1=plt.subplot(3,1,1)
	plt.xlim(0.999,1.001)
	plt.plot(xx,f(xx),label='$(1-x)^6$')
	plt.legend(loc='upper left',fontsize=14)
	

	ax2=plt.subplot(3,1,2)
	plt.xlim(0.999,1.001)
	plt.plot(xx,horner(xx),label='Hornerschema')
	plt.legend(loc='best',fontsize=14)

	ax3=plt.subplot(3,1,3)
	plt.xlim(0.999,1.001)
	plt.plot(xx,expanded(xx),label='Ausmultiplizert')
	plt.legend(loc='best',fontsize=14)
	
	plt.xlabel(r'x')
	plt.savefig("plot1.png")
	plt.show()

if __name__ == '__main__':
	aufg1()
