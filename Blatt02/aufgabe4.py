import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
def aufg4():
#b)
	rho = -0.8
	mean = [1,1]
	sigma_x = 0.2
	sigma_y = 0.2
	cov_xy = rho*sigma_y*sigma_x
	cov = [[sigma_x**2 , -cov_xy],[-cov_xy, sigma_y**2]]
	a0 = np.random.multivariate_normal(mean=mean,cov=cov,size=5000).T
	a1 = np.random.multivariate_normal(mean=mean,cov=cov,size=5000).T

	#df = pd.DataFrame(np.random.multivariate_normal(mean, cov, size=5000), columns=['a0', 'a1'])


	plt.scatter(a0,a1)
	plt.xlabel(r'$a_0$')
	plt.ylabel(r'$a_1$')
	plt.show()
	plt.close()
	def numerical(m,x,b):
		return m*x+b

	xx = np.linspace(-3,3,3)
	#print(xx)
	plt.plot(xx,numerical(a0,xx,a1),fmt='r--')
	plt.show()
	


if __name__ == '__main__':
	aufg4()



