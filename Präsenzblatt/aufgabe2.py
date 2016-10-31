import numpy as np 
import matplotlib.pyplot as plt
def aufg2():
	a = np.arange(100)+1
	print(a)
	b = np.array([a/100,
				np.random.random_sample((100,)),
				np.random.random_sample((100))*10,
				(np.random.random_sample((100,))+2)*10,
				np.random.normal(0,1,100),
				np.random.normal(5,2,100),
				(a/100)**2,
				np.cos(a/100)
				])
	#print(b)
	
	#plot 3b
	plt.plot(b[0],b[6],'ro',markersize=10)
	plt.xlabel('a/100')
	plt.ylabel('y')
	plt.grid(True)
	plt.title('3b')
	plt.savefig('3b.png',dpi=300)
	#plt.show()

	plt.clf()
	#plot 3c/d
	ax1=plt.subplot(1,2,1)
	l1=plt.plot(b[0],b[6],'ro',markersize=10,label=r'$a/100 ^2$')
	plt.legend(loc="best")

	ax2=plt.subplot(1,2,2)
	l2=plt.plot(b[0],b[7],'gs',markersize=10,label=r'$\cos(a/100)$')
	plt.legend(loc="best")
	plt.savefig('3c.png',dpi=300)
	plt.show()
if __name__ == '__main__':
	aufg2()