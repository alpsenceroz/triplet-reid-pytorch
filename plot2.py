import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_file_path.csv' with the path to your CSV file
file_path = '/home/bilginer/Downloads/losses (7).csv'

# Load the CSV file
data = pd.read_csv(file_path)

# Check the first few rows of the dataframe
print(data.head())

# Plotting reconstruction loss
plt.figure(figsize=(10, 6))
plt.plot(data['iteration'], data['loss_reconstruction'], label='Reconstruction Loss')
plt.xlabel('Iterations (Epochs)')
plt.ylabel('Loss')
plt.title('Reconstruction Loss vs. Iterations')
plt.legend()
plt.grid(True)
plt.show()

