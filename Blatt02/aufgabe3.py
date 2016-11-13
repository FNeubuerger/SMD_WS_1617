 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import cauchy
from scipy import stats  
plt.style.use('ggplot')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
plt.rcParams['font.family'] = 'lmodern'
def aufg3():


	
	'''	
	x_min = 0
	x_max = 1
	s = np.random.uniform(low=x_min, high=x_max, size=100)
	count_s, bins_s, ignored_s = plt.hist(s, 15, normed=True)
	plt.plot(bins_s, np.ones_like(bins_s), linewidth=2, color='r')
	#plt.show()
	#plt.close()

	e = np.random.exponential(scale=1.0, size=1000)
	count_e, bins_e, ignored_e = plt.hist(e, 15, normed=True)
	#plt.show()
	#plt.close()
	
	r = cauchy.rvs( size=1000)
	count_r, bins_r, ignored_r = plt.hist(r, bins=np.linspace (-10. , 10., 50), normed=True)
	#plt.show()
	#plt.close()
	'''
	data = np.load("empirisches_histogramm.npy")
	plt.hist(data['bin_mid'], bins=np.linspace (0. , 1., 50) ,
weights =data['hist'])
	plt.show()




	xmin, xmax = min(data['bin_mid']), max(data['bin_mid'])  
	lnspc = np.linspace(xmin, xmax, len(data['bin_mid']))
	m = np.mean(data['bin_mid'])
	s = np.std(data['bin_mid'])
	print(m,s)
	pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical 	values in our interval  
	
	plt.show()
	plt.plot(lnspc, pdf_g) # plot it
	plt.show()
if __name__ == '__main__':
	aufg3()
