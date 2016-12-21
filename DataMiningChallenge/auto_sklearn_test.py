#from __future__ import absolute_import, division, print_function
import autosklearn.regression
import autosklearn.classification
import sklearn.cross_validation
import sklearn.datasets
import sklearn.metrics

import tflearn

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

	digits = sklearn.datasets.load_digits()
	X = digits.data
	y = digits.target
	X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, random_state=1)
	automl = autosklearn.regression.AutoSklearnRegressor()
	automl.fit(X_train, y_train)
	y_hat = automl.predict(X_test)
	print("Models",automl.show_models())
	#print("Params", automl.get_params())
	print('Prediction', y_hat )
	print('Real_y',y_test)
	#output = pd.DataFrame(data = [y_hat], index=X_test.index.values)
	#output.to_csv('prediction.csv')
	return 0

def test_tflearn_regression():
	""" Linear Regression Example """

	

	# Regression data
	X = [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1]
	Y = [1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3]

	# Linear Regression graph
	input_ = tflearn.input_data(shape=[None])
	linear = tflearn.single_unit(input_)
	regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square',metric='R2', learning_rate=0.01)
	m = tflearn.DNN(regression)
	m.fit(X, Y, n_epoch=1000, show_metric=True, snapshot_epoch=False)

	print("\nRegression result:")
	print("Y = " + str(m.get_weights(linear.W)) +
	      "*X + " + str(m.get_weights(linear.b)))

	print("\nTest prediction for x = 3.2, 3.3, 3.4:")
	print(m.predict([3.2, 3.3, 3.4]))
	# should output (close, not exact) y = [1.5315033197402954, 1.5585315227508545, 1.5855598449707031]

def main():
	#test_automl_classification()
	#test_automl_regression()
	test_tflearn_regression()
	return 0

if __name__ == '__main__':
	main()