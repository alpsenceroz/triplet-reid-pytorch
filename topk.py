import os
import argparse
import random
from PIL import Image
from tqdm import tqdm

from torchvision import transforms
import torch

from modules import ResNet_VAE
from classifier import Classifier

BATCH_SIZE = 144
K = 6

transform = transforms.Compose([
    transforms.Resize((288, 144)),  # Resize the images
    transforms.ToTensor(),  # Convert the images to tensors
    transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225)),  # Normalize the images
])

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=3e-4, help='model weights directory')
    parser.add_argument('--classifier_dir', type=str, default=None, help='classifier weights')
    parser.add_argument('--dataset_dir', type=str, default=None, help='dataset directory')
    args = parser.parse_args()

    if args.dataset_dir is None:
        print('Please provide a dataset directory')
        exit()
    dataset_dir = args.dataset_dir
    model_dir = args.model_dir
    classifier_dir = args.classifier_dir

    # load model
    model = ResNet_VAE().cuda()
    classifier = Classifier(input_size=512).cuda()
    model.load_state_dict(torch.load(model_dir))
    classifier.load_state_dict(torch.load(classifier_dir))
    model.eval()
    classifier.eval()

    files = os.listdir(dataset_dir)
    files = [f for f in files if os.path.isfile(os.path.join(dataset_dir, f))] # list of files in dataset dir.

    random_file = random.choice(files) # Choose a random file

    anchor_id = int(random_file[:4])
    anchor_img_path = os.path.join(dataset_dir, random_file)

    print(f"Selected file: {random_file}")
    print(f"Selected ID: {anchor_id}")

    img = Image.open(anchor_img_path)
    img = transform(img)  # Apply the transformation
    img = img.unsqueeze(0)  # Add an extra dimension for the batch size
    img = img.cuda()

    _, anchor_embedding, _, _= model(img)
    anchor_embedding = anchor_embedding.repeat(BATCH_SIZE, 1)

    best_k = []
    best_k_filenames = []
    images = []
    filenames = []

    for idx, filename in enumerate(os.listdir(dataset_dir)):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Check if the file is an image
            img_path = os.path.join(dataset_dir, filename)
            img = Image.open(img_path)  # Open the image
            img_t = transform(img)  # Apply the transformation
            img_t = img_t.cuda()
            images.append(img_t)
            filenames.append(filename)

        if len(images) < BATCH_SIZE:
            continue
        images = torch.stack(images)

        with torch.no_grad():
            _, embeddings, _, _ = model(images)  # Get the embeddings
            embeddings = embeddings.view(embeddings.size(0), -1)  # Flatten the embeddings
            concatenated_embeddings = torch.cat((embeddings, anchor_embedding), dim=1)  # Concatenate the embeddings with the anchor embedding
            outputs = classifier(concatenated_embeddings)  # Get the outputs from the classifier
            outputs = outputs.view(-1)  # Flatten the outputs

            # Get the k largest outputs and their indices
            topk_values, topk_indices = torch.topk(outputs, K)

            # Convert the tensor to a list
            topk_values = topk_values.cpu().numpy().tolist()
            topk_indices = topk_indices.cpu().numpy().tolist()

            # Temporarily store the best scores and filenames from this batch
            current_batch_k_values = []
            current_batch_k_filenames = []
            for index in topk_indices:
                current_batch_k_values.append(outputs[index].item())
                current_batch_k_filenames.append(filenames[index])

            # Extend the best_k and best_k_filenames lists with new values
            best_k.extend(current_batch_k_values)
            best_k_filenames.extend(current_batch_k_filenames)

            # Sort by scores and slice to keep only the top K
            sorted_pairs = sorted(zip(best_k, best_k_filenames), reverse=True, key=lambda x: x[0])[:K]
            best_k, best_k_filenames = zip(*sorted_pairs)

            best_k = list(best_k)
            best_k_filenames = list(best_k_filenames)

        images = []  # Re-initialize after processing each batch
        filenames = []  # Re-initialize after processing each batch
        
    print(f"Best {K} values: {best_k}")
    print(f"Best {K} filenames: {best_k_filenames}")
    best_classes = [int(filename.split('_')[0]) for filename in best_k_filenames]
    matches = [x == anchor_id for x in best_classes]
    print(matches)
    
    

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import matplotlib.patches as patches


    anchor_image = mpimg.imread(anchor_img_path)

    # Create a new figure
    fig = plt.figure(figsize=(10, 10))

    # Add the anchor image to the figure
    ax = fig.add_subplot(1, K+1, 1)
    ax.imshow(anchor_image)
    ax.set_title("Anchor Image")
    ax.axis('off')

    # Add the top K images to the figure
    for i, filename in enumerate(best_k_filenames, start=2):
        img_path = os.path.join(dataset_dir, filename)
        img = mpimg.imread(img_path)
        ax = fig.add_subplot(1, K+1, i)
        ax.imshow(img)
        ax.set_title(f"Top {i-1}")
        ax.axis('off')
        
        # Add a colored border
        border_color = 'green' if matches[i-2] else 'red'
        rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, linewidth=2, edgecolor=border_color, facecolor='none')
        ax.add_patch(rect)
    
    # Create directory if it doesn't exist
    if not os.path.exists('./res'):
        os.makedirs('./res')
    plt.savefig('./res/topk.png')
    plt.show()