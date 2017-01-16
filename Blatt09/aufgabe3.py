import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
plt.style.use('ggplot')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
plt.rcParams['font.family'] = 'lmodern'
def aufg3():
	df = pd.read_csv('aufg_a.csv')
	print(df)
	return 0
	
if __name__ == '__main__':
	aufg3()
