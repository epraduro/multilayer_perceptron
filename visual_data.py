import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from utils import load_dataset_splited

def visualize_data(file_path="data.csv"):
    try:
        load_dataset_splited(file_path)
        df = pd.read_csv(file_path, header=None)
    except FileNotFoundError:
        print(f"The file {file_path} does not exist. Please provide a valid dataset.")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    columns = [
        'id',
        'diagnosis',
        'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness',
        'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
        'se_radius', 'se_texture', 'se_perimeter', 'se_area', 'se_smoothness',
        'se_compactness', 'se_concavity', 'se_concave_points', 'se_symmetry', 'se_fractal_dimension',
        'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness',
        'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension'
    ]

    df.columns = columns

    # Vérification rapide
    print(df.shape)
    print(df['diagnosis'].value_counts())
    print("\nPremières lignes :")
    print(df.head(3))

    # ───────────────────────────────
    # Graphique 1 : Répartition des classes
    # ───────────────────────────────
    plt.figure(figsize=(5,5))
    df['diagnosis'].value_counts().plot.pie(
        autopct='%1.1f%%',
        colors=['#ff9999', '#66b3ff'],
        startangle=90,
        textprops={'fontsize': 13}
    )
    plt.title("Répartition : Bénin (B) vs Malin (M)")
    plt.ylabel("")
    plt.show()

    # ───────────────────────────────
    # Graphique 2 : Scatter plot classique (2 features très discriminantes)
    # ───────────────────────────────
    plt.figure(figsize=(9,6))
    sns.scatterplot(data=df, x='worst_radius', y='worst_concave_points', hue='diagnosis', style='diagnosis', palette={'M': 'red', 'B': 'blue'}, alpha=0.5, s=20, edgecolor='black')
    plt.title("Worst Radius vs Worst Concave Points")
    plt.xlabel("Worst Radius")
    plt.ylabel("Worst Concave Points")
    plt.legend(title="Diagnostic")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        visualize_data(sys.argv[1])
    else:
        visualize_data()