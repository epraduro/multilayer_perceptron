import os
import numpy as np
import pandas as pd

def load_dataset_init(file_path: str):
	print(f"Loading dataset : {file_path}")

	if not file_path.endswith(".csv"):
		raise ValueError("The file must be in CSV format.")

	if not os.path.exists(file_path):
		raise FileNotFoundError(f"The file {file_path} does not exist.")

	data = pd.read_csv(file_path).values

	print(f"Dataset loaded successfully with shape {data.shape}")

	if data.shape[1] != 32:
		raise ValueError("The file must contain 32 columns (30 features + 1 label + 1 ID).")
	
	validated_data = []
	
	for row in data:
		if row.shape[0] != 32:
			raise ValueError("Each row must contain 32 elements.")
		if not isinstance(row[0], (int, float)):
			raise ValueError("The first column must be a number.")
		if row[1] not in ('B', 'M'):
			raise ValueError("The second column must be a string (B/M).")
		if not all(isinstance(x, (int, float)) for x in row[2:32]):
			raise ValueError("The first 30 columns must be numbers.")
		validated_data.append(row)

	return True

def load_dataset_splited(file_path: str):
	print(f"Loading dataset : {file_path}")

	if not file_path.endswith(".csv"):
		raise ValueError("The file must be in CSV format.")

	if not os.path.exists(file_path):
		raise FileNotFoundError(f"The file {file_path} does not exist.")

	data = pd.read_csv(file_path).values

	if data.shape[1] != 31:
		raise ValueError("The file must contain 31 columns (30 features + 1 label).")
	
	validated_data = []
	
	for row in data:
		if row.shape[0] != 31:
			raise ValueError("Each row must contain 31 elements.")
		if not all(isinstance(x, (int, float)) for x in row[:30]):
			raise ValueError("The first 30 columns must be numbers.")
		if row[-1] not in ('B', 'M'):
			raise ValueError("The last column must be a string (B/M).")
		validated_data.append(row)

	return True


def sigmoid(z):
	z = np.clip(z, -500, 500)
	return 1 / (1 + np.exp(-z))