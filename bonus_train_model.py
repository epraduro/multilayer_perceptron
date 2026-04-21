import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from train_model import train
import sys
import argparse
from utils import load_dataset_splited


def bonus():
    plt.figure(figsize=(11, 7))

    if not load_dataset_splited("breast_cancer_train.csv"):
        raise ValueError("The training dataset is not valid.")
    if not load_dataset_splited("breast_cancer_val.csv"):
        raise ValueError("The validation dataset is not valid.")
    train_data = "breast_cancer_train.csv"
    val_data = "breast_cancer_val.csv"

    configs = [
        {"name": "2_hidden", "layer_sizes": [30, 16, 8, 1], "lr": 0.0005},
        {"name": "3_hidden", "layer_sizes": [30, 32, 16, 8, 1], "lr": 0.0005},
        {"name": "deep", "layer_sizes": [30, 64, 32, 16, 8, 4, 1], "lr": 0.0005},
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))

    for i, cfg in enumerate(configs):
        color = colors[i]
        train_loss, val_loss, _, _ = train(
            train_data=train_data,
            val_data=val_data,
            config_name=cfg["name"],
            layer_sizes=cfg["layer_sizes"],
            learning_rate=cfg["lr"],
            epochs=300,
        )

        plt.plot(train_loss, label=f'{cfg["name"]} - Train', color=color, linestyle='--', linewidth=1.8)
        plt.plot(val_loss,   label=f'{cfg["name"]} - Val',   color=color, linestyle='-',  linewidth=2.2)


    plt.title("Comparison of Learning Curves with Different Architectures")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (log scale)")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"learning_curves_comparison_{timestamp}.png", dpi=300)
    plt.show()
    

if __name__ == "__main__":  
    bonus()