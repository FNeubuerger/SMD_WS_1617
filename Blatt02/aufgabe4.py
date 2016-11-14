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
	cov = [[sigma_x**2 , cov_xy],[cov_xy, sigma_y**2]]
	a = np.random.multivariate_normal(mean=mean,cov=cov,size=100000)

	#df = pd.DataFrame(np.random.multivariate_normal(mean, cov, size=5000), columns=['a0', 'a1'])


	plt.scatter(a[:,0],a[:,1],rasterized=True)
	plt.xlabel(r'$a_0$')
	plt.ylabel(r'$a_1$')
	plt.savefig('scatterplot_a0_a1.png',dpi=300)
	plt.show()
	plt.close()
	
	y = [ a[i,0] + a[i,1]*(-3) for i in range(len(a))]
	std_1 = np.std(y)

	y = [ a[i,0] + a[i,1]*(0) for i in range(len(a))]
	std_2 = np.std(y)

	y = [ a[i,0] + a[i,1]*(3) for i in range(len(a))]
	std_3 = np.std(y)

	print('-3: ',std_1)
	print('0: ',std_2)
	print('3: ',std_3)

if __name__ == '__main__':
	aufg4()



