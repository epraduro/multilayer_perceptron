import pandas as pd
import numpy as np
import argparse
from utils import load_dataset_splited
from utils import sigmoid
import matplotlib.pyplot as plt
import datetime

import os

def train(train_data, val_data, epochs=100, learning_rate=0.0005, seed=42, batch_size=8, layer_sizes=[30, 16, 1], config_name="1"):
	train_df = pd.read_csv(train_data)
	val_df = pd.read_csv(val_data)

	X_train = train_df.drop(columns=["diagnosis"]).values
	y_train = train_df["diagnosis"].map({"B": 0, "M": 1}).values.reshape(-1, 1)
	X_val = val_df.drop(columns=["diagnosis"]).values
	y_val = val_df["diagnosis"].map({"B": 0, "M": 1}).values.reshape(-1, 1)

	### normalization of the data (using train set statistics) and save mean/std for later use in predict.py

	mean = X_train.mean(axis=0)
	std = X_train.std(axis=0) + 1e-8
	X_train = ( X_train - mean ) / std
	X_val = ( X_val - mean ) / std

	### hyperparameters
	n_layers = len(layer_sizes) - 1


	### initiate model parameters (weights and biases) or load existing model if available
	if os.path.exists(f"save_model_{config_name}.npy"):
		print("Chargement du modèle existant...")
		modele = np.load(f"save_model_{config_name}.npy", allow_pickle=True).item()
		weights = modele["weights"]
		biases = modele["biases"]
		mean, std = modele["mean"], modele["std"]
	else:
		print("Initialisation du modèle...")
		np.random.seed(seed)
		weights = []
		biases = []
		for i in range(n_layers):
			scale = np.sqrt(2.0 / layer_sizes[i])
			W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
			b = np.zeros((1, layer_sizes[i+1]))
			weights.append(W)
			biases.append(b)

	### Training loop

	# Listes pour les courbes
	train_losses = []
	val_losses = []

	accuracy_train = []
	accuracy_val = []

	for epoch in range(epochs):
		# === Training phase ===
		epoch_loss = 0.0
		num_batches = 0
		
		# Shuffle the training data at the beginning of each epoch
		indices = np.random.permutation(len(X_train))
		X_train_shuffled = X_train[indices]
		y_train_shuffled = y_train[indices]

		for start in range(0, len(X_train), batch_size):
			end = start + batch_size
			X_batch = X_train_shuffled[start:end]
			y_batch = y_train_shuffled[start:end]

			# Forward pass
			activations = [X_batch]
			for i in range(n_layers):
				z = activations[-1] @ weights[i] + biases[i]
				a = sigmoid(z)
				activations.append(a)

			A_last = activations[-1]

			# Calculate loss for the batch
			loss_train = -np.mean(y_batch * np.log(A_last + 1e-15) + 
						(1 - y_batch) * np.log(1 - A_last + 1e-15))
			epoch_loss += loss_train
			num_batches += 1

			# Backward pass
			dz = A_last - y_batch
			dweights = []
			dbiases = []

			for i in range(n_layers-1, -1, -1):
				dW = activations[i].T @ dz
				db = np.sum(dz, axis=0, keepdims=True)
				dweights.append(dW)
				dbiases.append(db)

				if i > 0:
					dz = (dz @ weights[i].T) * activations[i] * (1 - activations[i])

			dweights.reverse()
			dbiases.reverse()

			# Update weights and biases
			for i in range(n_layers):
				weights[i] -= learning_rate * dweights[i]
				biases[i] -= learning_rate * dbiases[i]

		# === Validation phase ===
		activations_val = [X_val]
		for i in range(n_layers):
			z_val = activations_val[-1] @ weights[i] + biases[i]
			a_val = sigmoid(z_val)
			activations_val.append(a_val)
		
		A_last_val = activations_val[-1]
		loss_val = -np.mean(y_val * np.log(A_last_val + 1e-15) + 
						(1 - y_val) * np.log(1 - A_last_val + 1e-15))
		val_losses.append(loss_val)

		avg_train_loss = epoch_loss / num_batches
		train_losses.append(avg_train_loss)

		# Calculate accuracy on entire training set
		activations_train_full = [X_train]
		for i in range(n_layers):
			z_train = activations_train_full[-1] @ weights[i] + biases[i]
			a_train = sigmoid(z_train)
			activations_train_full.append(a_train)
		A_last_train = activations_train_full[-1]
		acc_train = ((A_last_train > 0.5) == y_train).mean()
		accuracy_train.append(acc_train)
		
		acc_val = ((A_last_val > 0.5) == y_val).mean()
		accuracy_val.append(acc_val)

		if (epoch + 1) % 10 == 0 or epoch == 0:
			print(f"Epoch {epoch+1:3d}/{epochs} - Loss train: {avg_train_loss:.4f} | Loss val: {loss_val:.4f} | Acc train: {acc_train:.4f} | Acc val: {acc_val:.4f}")
			


	### Saving the model

	modele = {"weights": weights, "biases": biases, "mean": mean, "std": std}
	np.save(f"save_model_{config_name}.npy", modele)
	print(f"Saving model to './save_model_{config_name}.npy'...")

	return train_losses, val_losses, accuracy_train, accuracy_val

def main():
	
	parser = argparse.ArgumentParser(prog="train_model", description="Train a simple MLP on the breast cancer dataset")
	parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of training epochs")
	parser.add_argument("-lr", "--learning_rate", type=float, default=0.0005, help="Learning rate for training")
	parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for reproducibility")
	parser.add_argument("-b", "--batch_size", type=int, default=8, help="Batch size for training")
	parser.add_argument("-l", "--layers", type=int, nargs="+", default=[30, 16, 1], help="Sizes of each layer in the MLP (including input and output layers)")
	parser.add_argument("-t", "--train_data", type=str, default="breast_cancer_train.csv", help="Path to the training dataset (CSV file)")
	parser.add_argument("-v", "--val_data", type=str, default="breast_cancer_val.csv", help="Path to the validation dataset (CSV file)")
	args = parser.parse_args()

	try:
		load_dataset_splited(args.train_data)
		load_dataset_splited(args.val_data)
	except FileNotFoundError:
		print("The training or validation dataset does not exist. Please provide valid datasets.")
		return
	except Exception as e:
		print(f"Error loading dataset: {e}")
		return
	
	train_data = args.train_data
	val_data = args.val_data

	train_losses, val_losses, accuracy_train, accuracy_val = train(
		train_data,
		val_data,
		epochs=args.epochs,
		learning_rate=args.learning_rate,
		seed=args.seed,
		batch_size=args.batch_size,
		layer_sizes=args.layers
	)

	### Plotting learning curves
	plt.figure(figsize=(8, 4))
	plt.plot(train_losses, label='Train Loss', color='blue', linestyle='--', linewidth=1.3)
	plt.plot(val_losses, label='Validation Loss', color='orange', linewidth=1.6)
	plt.title("Learning Curves")
	plt.xlabel("Epochs")
	plt.ylabel("Loss (log scale)")
	plt.yscale('log')
	plt.legend()
	plt.tight_layout()
	plt.savefig(f"learning_curves_train_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
	print(f"Learning curves saved to 'learning_curves_train_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png'")

	### Plotting accuracy curves
	plt.figure(figsize=(8, 4))
	plt.plot(accuracy_train, label='Train Accuracy', color='blue', linestyle='--', linewidth=1.3)
	plt.plot(accuracy_val, label='Validation Accuracy', color='orange', linewidth=1.6)
	plt.title("Learning Curves")
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy")
	plt.yscale('linear')
	plt.legend()
	plt.tight_layout()
	plt.savefig(f"learning_curves_accuracy_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
	print(f"Learning curves saved to 'learning_curves_accuracy_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png'")

if __name__ == "__main__":
	main()
