 
import numpy as np 
import matplotlib.pyplot as plt 
import numpy as np
# Import des Modules numpy als np
import scipy as sp
# Import des Modules scipy als sp
import matplotlib.pyplot as plt # Import als plt

plt.style.use('ggplot')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
plt.rcParams['font.family'] = 'lmodern'
def aufg3():

	a = np.array([-0.368,0.441,-1.177,-3.530])
	b = [-2.942,-4.707,-6.472]


	x = np.linspace(-8, 1,10)
	a_x = np.array([0,0,0,0])
	b_x = np.array([0,0,0])

	y = np.array([0,0,0,0,0,0,0,0,0,0]) # gibt 50 Zahlen in gleichmäßigem Abstand von 0–1
	plt.plot(a,a_x,'ro', label='$P_0$')
	plt.plot(b,b_x,'go', label='$P_1$')
	plt.legend()
	plt.plot(x, y)
	plt.savefig('aufg3.png')
	plt.show()














if __name__ == '__main__':
	aufg3()
