import autosklearn.regression
import autosklearn.classification
import sklearn.cross_validation
import sklearn.datasets
import sklearn.metrics
import pandas as pd 
import numpy as np
def generate_features(train_data,test_data):
	### generates features on training and test set

	return train , test


def read_data(train_path,test_path,gen_features=False):
	#reads data from csv file into pandas dataframe
	#returns training and test set

	train = pd.read_csv(train_path)
	test = pd.read_csv(test_path)
	if gen_features == True:
		train,test = generate_features(train,test)

	return train,test


def test_automl_classification():
	digits = sklearn.datasets.load_digits()
	print(digits)
	X = digits.data
	y = digits.target
	X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, random_state=1)
	automl = autosklearn.classification.AutoSklearnClassifier()
	automl.fit(X_train, y_train)
	y_hat = automl.predict(X_test)
	print("Models",automl.show_models())
	#print("Params", automl.get_params())
	print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))
	return 0

def test_automl_regression():

	#digits = sklearn.datasets.load_digits()
	#X = digits.data
	#y = digits.target
	#X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, random_state=1)
	train, X_test = read_data('train.csv','test.csv') 
	y_train = train['SalePrice'].values
	print(y_train)
	X_train=train.values
	print(X_train)
	automl = autosklearn.regression.AutoSklearnRegressor()
	automl.fit(X_train, y_train)
	y_hat = automl.predict(X_test)
	print("Models",automl.show_models())
	#print("Params", automl.get_params())
	print('Prediction', y_hat )
	#print('Real_y',y_test)
	#output = pd.DataFrame(data = [y_hat], index=X_test.index.values)
	#output.to_csv('prediction.csv')
	return 0


def main():
	test_automl_regression()
	
	return 0

if __name__ == '__main__':
	main()