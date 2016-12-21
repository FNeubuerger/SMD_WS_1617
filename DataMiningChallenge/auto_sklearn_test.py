import autosklearn.regression
import autosklearn.classification
import sklearn.cross_validation
import sklearn.datasets
import sklearn.metrics

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


def main():
	test_automl_classification()
	
	return 0

if __name__ == '__main__':
	main()