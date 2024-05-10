import os
import os
import shutil
import random

# Define the folder paths
source_folder = "./datasets/Market-1501-v15.09.15/bounding_box_test"
validation_folder = "./datasets/Market-1501-v15.09.15/bounding_box_validation"
test_folder = "./datasets/Market-1501-v15.09.15/bounding_box_test"

# Create the validation and test folders if they don't exist
os.makedirs(validation_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Initialize dictionaries to store file paths for each class
class_files = {}

# Iterate through all the files in the source folder
for filename in os.listdir(source_folder):
    # Extract the class name from the file name
    class_name = filename.split('_')[0]
    
    # Add the file path to the respective class in the dictionary
    if class_name in class_files:
        class_files[class_name].append(os.path.join(source_folder, filename))
    else:
        class_files[class_name] = [os.path.join(source_folder, filename)]

# Iterate through each class
for class_name, files in class_files.items():
    # Calculate the number of files to move to validation (40% of total)
    num_validation_files = int(0.4 * len(files))
    
    # Randomly select files for validation
    validation_files = random.sample(files, num_validation_files)
    
    # Move validation files to the validation folder
    for file in validation_files:
        destination = os.path.join(validation_folder, os.path.basename(file))
        shutil.move(file, destination)
    
    # Move remaining files to the test folder
    for file in files:
        if file not in validation_files:
            destination = os.path.join(test_folder, os.path.basename(file))
            shutil.move(file, destination)

print(f"Splitting complete. Validation split created in {validation_folder} folder, \
      and test split in {test_folder} folder.")

# Define the folder path
folder_path = validation_folder

# Initialize a dictionary to store the counts for each class
class_counts = {}

# Iterate through all the files in the folder and count instances for each class
for filename in os.listdir(folder_path):
    # Extract the class name from the file name
    class_name = filename.split('_')[0]
    
    # Update the count for the class in the dictionary
    if class_name in class_counts:
        class_counts[class_name] += 1
    else:
        class_counts[class_name] = 1

# Print the class counts
for class_name, count in class_counts.items():
    
    # If class count is less than 2, remove the image files
    if count < 2:
        # Get the file names for the class
        class_files = [filename for filename in os.listdir(folder_path) if filename.startswith(class_name)]
        print(f"removing class {class_name} with count {count}")
        # Remove each file
        for file_name in class_files:
            os.remove(os.path.join(folder_path, file_name))

print(f"Images with class count less than 2 have been removed in {folder_path}.")

