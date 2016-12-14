import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

plt.style.use('ggplot')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True
plt.rcParams['font.family'] = 'lmodern'
def aufg2():



	def labelling(exa,label):
  #returns labelled set
		if label==1:
			exa['label'] = np.ones(len(exa))
		elif label ==0:
			exa['label'] = np.zeros(len(exa))
		return exa
	P0 = np.load('P0.npy')
	P1 = np.load('P1.npy')

	P0 = pd.DataFrame(P0).T
	P1 = pd.DataFrame(P1).T

	P0=labelling(P0,0)
	P1=labelling(P1,1)
	result = pd.concat([P0,P1], ignore_index=True)
	result.columns = ['x','y', 'label']




	def sigmoid(x):
		output = 1/(1+np.exp(-x))
		return output

	def sigmoid_output_to_derivative(output):
		return output*(1-output)

	X = result[['x','y']]
	y = result['label']
	X = np.array(X)
	y = np.array(y)


	alpha,hidden_dim = (0.5,1)
	synapse_0 = 2*np.random.random((2,hidden_dim)) - 1

	for j in range(5):
		layer_1 = 1/(1+np.exp(-(np.dot(X,synapse_0))))
		#layer_2 = 1/(1+np.exp(-(np.dot(layer_1,synapse_1))))
		#layer_2_delta = (layer_2 - y)*(layer_2*(1-layer_2))
		#layer_1_delta = layer_2_delta.dot(synapse_1.T) * (layer_1 * (1-layer_1))
		#synapse_1 -= (alpha * layer_1.T.dot(layer_2_delta))
		#synapse_0 -= (alpha * X.T.dot(layer_1_delta))

		layer_0 = X
		layer_1 = sigmoid(np.dot(layer_0,synapse_0))
		layer_1_error = layer_1 - y
		layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
		synapse_0_derivative = np.dot(layer_0.T,layer_1_delta)
		synapse_0 -= synapse_0_derivative
		print("Output After Training:")
		print(layer_1)
if __name__ == '__main__':
	aufg2()
