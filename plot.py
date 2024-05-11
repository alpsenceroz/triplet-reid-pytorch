import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_file_path.csv' with the path to your CSV file
file_path = '/home/bilginer/Downloads/losses (11).csv'

# Load the CSV file
data = pd.read_csv(file_path)

# Check the first few rows of the dataframe
print(data.head())

# Plotting val_loss and training_loss_avg
plt.figure(figsize=(10, 6))
plt.plot(data['iteration'], data['val_loss'], label='Validation Loss')
plt.plot(data['iteration'], data['loss_bce'], label='Training Loss Average')
plt.xlabel('Iterations (Epochs)')
plt.ylabel('Loss')
plt.title('Validation Loss and Training Loss Average vs. Iterations')
plt.legend()
plt.grid(True)
plt.show()

