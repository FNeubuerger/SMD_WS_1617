 
import numpy as np 
import matplotlib.pyplot as plt 

plt.style.use('ggplot')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
plt.rcParams['font.family'] = 'lmodern'
def aufg2():
	P0 = np.load('P0.npy')
	P1 = np.load('P1.npy')

	npa = np.array
	
	def softmax(w, t = 1.0):
		e = np.exp(npa(w)/t)
		dist = e/np.sum(e)
		return dist

	w = np.array([0.1,0.2])
	print(softmax(w))

if __name__ == '__main__':
	aufg2()
