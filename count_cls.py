import os

# Define the folder path
folder_path = "./datasets/Market-1501-v15.09.15/bounding_box_test"

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
    # If class count is less than 4, remove the image files
    if count < 4:
        # Get the file names for the class
        # class_files = [filename for filename in os.listdir(folder_path) if filename.startswith(class_name)]
        print(f"Class {class_name}: {count}")
        # Remove each file
        # for file_name in class_files:
            # os.remove(os.path.join(folder_path, file_name))
