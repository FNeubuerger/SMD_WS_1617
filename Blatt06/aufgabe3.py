 
import numpy as np 
import matplotlib.pyplot as plt 

plt.style.use('ggplot')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
plt.rcParams['font.family'] = 'lmodern'
def aufg3():
	##########################
	###Definiere Funktionen###
	##########################
	def f(x):
		return (x**3 + 1/3) - (x**3 - 1/3)

	def g(x):
		return ((3 + (x**3)/3) - (3 - (x**3)/3))/x**3

	xx = np.logspace(-20,20,41)

	def f_alg(x): #grenzwert der Funktion
		return 2/3 +0*x

	print('für diese Werte weicht f(x) nur um 1% vom algebraischen Wert ab')
	for i in xx:
		if abs((f(i)-f_alg(i))/f_alg(i)) <=0.01:
			print(i)
	print('für diese Werte weicht g(x) nur um 1% vom algebraischen Wert ab')
	for i in xx:
		if abs((g(i)-f_alg(i))/f_alg(i)) <=0.01:
			print(i)

	print('für diese Werte ist f(x) = 0')
	for i in xx:
		if f(i)==0:
			print(i)

	print('für diese Werte ist g(x) = 0')
	for i in xx:
		if g(i)==0:
			print(i)

	##########################
	#####Plotte Funktionen####
	##########################


	fig = plt.figure(figsize=(16,9))
	plt.plot(xx,f(xx),'rx',markersize=10,label=r'$f(x)$')
	plt.plot(xx,f_alg(xx),'k--',label=r'vereinfacht $f(x) = g(x) = 2/3$')
	plt.plot(xx,g(xx),'bo',label=r'$g(x)$')
	plt.rcParams.update({'font.size': 22})
	plt.xlabel(r'x')
	plt.ylabel(r'Wert der Funktion')
	plt.xscale('log')
	#plt.yscale('log')
	#plt.xlim(-1e-2,1e20)
	plt.ylim(-1e-2 , 0.9)
	plt.legend(loc='best')
	plt.savefig("plot3.png")
	plt.show()

if __name__ == '__main__':
	aufg3()
