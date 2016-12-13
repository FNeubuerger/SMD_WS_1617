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


# compute sigmoid nonlinearity

	def sigmoid(x):
		output = 1/(1+np.exp(-x))
		return output

# convert output of sigmoid function to its derivative
	def sigmoid_output_to_derivative(output):
		return output*(1-output)

# input dataset
	X = result[['x','y']]

	y = result['label']

# seed random numbers to make calculation
# deterministic (just a good practice)
	np.random.seed(1)

# initialize weights randomly with mean 0

	synapse_0 = 2*np.random.random((2,1)) - 1


	for iter in range(2):
# forward propagation
		layer_0 = X
		layer_1 = sigmoid(layer_0.dot(synapse_0))
		layer_1_error = layer_1.subtract(y)

# multiply how much we missed by the
# slope of the sigmoid at the values in l1
		layer_1_delta=layer_1_error*sigmoid_output_to_derivative(layer_1)

		synapse_0_derivative = np.dot(layer_0.T,layer_1_delta)

# update weights

		synapse_0 -= synapse_0_derivative

		print("Output After Training:")

		print(layer_1)
if __name__ == '__main__':
	aufg2()
