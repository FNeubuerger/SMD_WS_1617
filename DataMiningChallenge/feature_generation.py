import pandas as pd 
import numpy as np 


def generate_features():
	train = pd.read_csv('train.csv')
	test = pd.read_csv('test.csv')
	target = train['SalePrice']
	print(target)


if __name__ == '__main__':
	generate_features()