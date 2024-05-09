# -*- coding: utf-8 -*-
"""
Created on Thu May  9 17:27:04 2024

@author: murat
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_results(csv_file): # need to input filename or we need to create a bash file for that too
    data = pd.read_csv(csv_file)

    plt.figure(figsize=(10, 6))
    plt.plot(data['Iteration'], data['Loss'], label='Loss')
    plt.title('Training Loss over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(data['Iteration'], data['Learning Rate'], label='Learning Rate', color='r')
    plt.title('Learning Rate over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_rate.png')
    plt.show()

def main():
    csv_file = './res/training_log.csv'
    plot_training_results(csv_file)

if __name__ == "__main__":
    main()

# for the heat map #

"""
def plot_heatmap(data_array, title='Heatmap Title', xlabel='X-axis Label', ylabel='Y-axis Label'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(data_array, annot=True, fmt="f", cmap='coolwarm')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
"""