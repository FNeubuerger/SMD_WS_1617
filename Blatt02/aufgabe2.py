 
import numpy as np 
import matplotlib.pyplot as plt 

plt.style.use('ggplot')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
plt.rcParams['font.family'] = 'lmodern'

def rd_lk(a,b,m,seed=0):
	x_a = seed #whatever seed value default 0
	arr = [] #empty list for storage

	for i in range(m): #loop for generation
		x_n =(a*x_a+b)%m #generates linear kongruent random numbers
		arr.append(x_n/m) #append normed values
		x_a = x_n #set new seed

	return arr #returns array of normed random numbers 


def aufg2():
	a = rd_lk(a=1601,b=3456,m=10000)
	#print(a)
	fig = plt.figure(figsize=(16,9))
	plt.hist(a,bins=100,rasterized=True)
	plt.xlabel(r'random number value')
	plt.ylabel(r'n')
	plt.savefig('linear_kongruent_random_numbers.png',dpi=300)
	plt.show()


if __name__ == '__main__':
	aufg2()
