import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
def aufg4():
#b)
	mean = [1,1]
	cov = [[0.2**2,-0.4],[-0.4,0.2**2]]
	a0 = np.random.multivariate_normal(mean=mean,cov=cov,size=5000).T
	a1 = np.random.multivariate_normal(mean=mean,cov=cov,size=5000).T

	#df = pd.DataFrame(np.random.multivariate_normal(mean, cov, size=5000), columns=['a0', 'a1'])


	plt.scatter(a0,a1)
	plt.xlabel(r'$a_0$')
	plt.ylabel(r'$a_1$')
	#plt.show()
	plt.close()
	def numerical(m,x,b):
		return m*x+b

	xx = np.linspace(-3,3,3)
	#print(xx)
	plt.plot(xx,numerical(a0,xx,a1),fmt='r--')
	plt.show()
	


if __name__ == '__main__':
	aufg4()



