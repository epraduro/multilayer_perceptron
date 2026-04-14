import pandas as pd
import numpy as np
import sys
from utils import sigmoid
from utils import load_dataset

def load_model(model_file):
	try:
		modele = np.load(model_file, allow_pickle=True).item()
		weights = modele["weights"]
		biases = modele["biases"]
		mean, std = modele["mean"], modele["std"]

		print("Model loaded successfully.")
		return weights, biases, mean, std
	except FileNotFoundError:
		print("No model found. Please train the model before making predictions.")
		exit(1)

def predict(data_file, model_file):
	if len(sys.argv) < 3:
		print("Usage : python3 predict.py <data_file.csv> <model_file.npy>")
		exit(1)

	df = pd.read_csv(data_file)
	X = df.drop(columns=["diagnosis"]).values
	y_true = df["diagnosis"].map({"B": 0, "M": 1}).values.reshape(-1, 1)

	weights, biases, mean, std = load_model(model_file)

	X = ( X - mean ) / std

	activations = [X]
	for W, b in zip(weights, biases):
		Z = activations[-1] @ W + b
		A = sigmoid(Z)
		activations.append(A)

	A_last = activations[-1]

	predictions = (A_last >= 0.5).astype(int)

	eps = 1e-15
	loss = -np.mean(y_true * np.log(A_last + eps) + (1 - y_true) * np.log(1 - A_last + eps))
	accuracy = (predictions == y_true).mean()

	print(f"Loss on the set : {loss:.4f}")
	print(f"Accuracy on the set : {accuracy:.4f}")

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("Usage : python3 predict.py <data_file.csv> <model_file.npy>")
		exit(1)
	elif not load_dataset(sys.argv[1]):
		print("data_file.csv is not valid.")
		exit(1)
	elif not sys.argv[2].endswith(".npy"):
		print("The model must be a .npy file.")
		exit(1)
	else:
		predict(sys.argv[1], sys.argv[2])