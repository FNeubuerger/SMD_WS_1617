import pandas as pd 
import numpy as np 
import autosklearn.regression
import autosklearn.classification
import sklearn.cross_validation
import sklearn.datasets
import sklearn.metrics

def generate_features(train_data,test_data):
	### generates features on training and test set

	return train , test


def read_data(train_path,test_path,gen_features=False):
	#reads data from csv file into pandas dataframe
	#returns training and test set

	train = pd.read_csv(train_path)
	test = pd.read_csv(test_path)
	if gen_features == True:
		generate generate_features(train,test)

	return train,test


def regression_automl(train_data,test_data):
	#without cross validation
	X_train = train_data#.drop(['SalePrice']) #data without target
	y_train = train_data['SalePrice'] #target
	X_test = test_data#.drop(['SalePrice'])
	automl = autosklearn.regression.AutoSklearnRegressor()
	automl.fit(X_train, y_train)
	prediction = automl.predict(X_test)
	print("Params", automl.get_params())
	print("Models",automl.show_models())
	return 0

def best_regression(train_data,test_data)

def main():
	train,test = read_data('train.csv','test.csv',gen_features=False)
	regression_automl()



if __name__ == '__main__':
	main()