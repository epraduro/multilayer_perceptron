import numpy as np
import pandas as pd
import sys
from utils import load_dataset_init

def split_and_save(X, y, train_ratio=0.8, random_seed=42, train_file="train.csv", val_file="val.csv"):
    np.random.seed(random_seed)
    
    mask_b = (y == 'B')
    mask_m = (y == 'M')
    
    idx_b = np.where(mask_b)[0]
    idx_m = np.where(mask_m)[0]
    
    np.random.shuffle(idx_b)
    np.random.shuffle(idx_m)
    
    train_size_b = int(train_ratio * len(idx_b))
    train_size_m = int(train_ratio * len(idx_m))
    
    train_idx_b = idx_b[:train_size_b]
    val_idx_b   = idx_b[train_size_b:]
    
    train_idx_m = idx_m[:train_size_m]
    val_idx_m   = idx_m[train_size_m:]
    
    train_idx = np.concatenate([train_idx_b, train_idx_m])
    val_idx   = np.concatenate([val_idx_b,   val_idx_m])
    
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val   = X.iloc[val_idx]
    y_val   = y.iloc[val_idx]
    
    df_train = X_train.copy()
    df_train['diagnosis'] = y_train.values
    
    df_val = X_val.copy()
    df_val['diagnosis'] = y_val.values
    
    df_train.to_csv(train_file, index=False)
    df_val.to_csv(val_file, index=False)
    
    print(f"Saved : {train_file:20} ({len(df_train):4d} lines )")
    print(f"Saved : {val_file:20}   ({len(df_val):4d} lines )")
    print("\nDistribution train :", y_train.value_counts(normalize=True).round(4))
    print("\nDistribution val   :", y_val.value_counts(normalize=True).round(4))
    
    print("\nExample train :")
    print(df_train)
    print("\nExample val :")
    print(df_val)

    return df_train, df_val


def split_data():
    if len(sys.argv) != 2:
        print("Usage: python3 split_data.py <path_to_data.csv>")
        sys.exit(1)

    try:
        load_dataset_init(sys.argv[1])
        df = pd.read_csv(sys.argv[1], header=None)
    except FileNotFoundError:
        print("The file data.csv does not exist. Please provide a valid dataset.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    columns = [
        'id', 'diagnosis',
        'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness',
        'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
        'se_radius', 'se_texture', 'se_perimeter', 'se_area', 'se_smoothness',
        'se_compactness', 'se_concavity', 'se_concave_points', 'se_symmetry', 'se_fractal_dimension',
        'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness',
        'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension'
    ]

    df.columns = columns

    X = df.drop(columns=['id', 'diagnosis'])
    y = df['diagnosis']

    split_and_save(
        X, y,
        train_ratio=0.8,
        random_seed=42,
        train_file="breast_cancer_train.csv",
        val_file="breast_cancer_val.csv"
    )

if __name__ == "__main__":
    split_data()