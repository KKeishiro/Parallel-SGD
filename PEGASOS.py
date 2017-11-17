import numpy as np
import random

m = 10000
sigma = 4
nround = 100000
lambda_val = 0.01

def transform(X):
	np.random.seed(123)
	new_X = []
	w = np.random.multivariate_normal(mean = np.zeros(X.shape[1]), cov = np.identity(X.shape[1]) * sigma**2, size = m)
	b = np.random.uniform(0, 2 * np.pi, m)
	X_new = np.sqrt(2.0 / m) * np.cos(np.dot(X, np.transpose(w)) + b)
	X_new = (X_new - np.mean(X_new, 0)) / np.std(X_new, 0)
	return X_new

def mapper(key, value):
	Y = []
	X = []

	for i in range(len(value)):
		feature = value[i].split()[1:]
		label = value[i].split()[0]
		X.append([float(m) for m in feature])
		Y.append(float(label))

	X = np.array(X)
	X = transform(X)
	Y = np.array(Y)

	n_features = X.shape[1]
	w = np.zeros(n_features)

	for t in range(1, nround):
		i = np.random.randint(0, X.shape[0])
		learning_rate = 1.0 / (lambda_val * t)
		if Y[i] * np.dot(X[i], w) < 1:
			w = (1 - learning_rate * lambda_val) * w + learning_rate * Y[i] * X[i]
		else:
			w = (1 - learning_rate * lambda_val) * w
		w = w * min(1, 1 / (np.sqrt(lambda_val) * np.linalg.norm(w)))
	yield 1, w

def reducer(key, values):
	yield np.mean(values, axis=0)
