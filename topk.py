import os
import argparse
import random
from PIL import Image
from tqdm import tqdm

from torchvision import transforms
import torch

import utils

BATCH_SIZE = 144
K = 6

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone_dir', type=str, default=None, help='backbone weights directory')
    parser.add_argument('--backbone_type', type=str, default=None, help='backbone name')
    parser.add_argument('--classifier_dir', type=str, default=None, help='Autoencoder weights')
    parser.add_argument('--ae_dir', type=str, default=None, help='autoencoder weights')
    parser.add_argument('--ae_type', type=str, default=None, help='autoencoder type')
    parser.add_argument('--dataset_dir', type=str, default=None, help='dataset directory')
    args = parser.parse_args()

    if args.dataset_dir is None:
        print('Please provide a dataset directory')
        exit()
    if args.ae_dir is None or args.ae_type is None:
        print('Please provide an autoencoder for evaluation')
        exit()
    if args.classifier_dir is None:
        print('Please provide a classifier for evaluation')
        exit()
    if args.backbone_dir is None or args.backbone_type is None:
        print('Please provide a backbone for evaluation')
        exit()

    backbone_type = args.backbone_type
    backbone_dir = args.backbone_dir
    classifier_dir = args.classifier_dir
    ae_dir = args.ae_dir
    ae_type = args.ae_type
    dataset_dir = args.dataset_dir

    if backbone_type == 'swin':
        transform = transforms.Compose([
            transforms.Resize((224, 112)),
            transforms.Pad((56, 0, 56, 0), fill=0),
            transforms.ToTensor(),
            transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225)),
        ])

    backbone, ae, classifier = utils.load_model(backbone_type, backbone_dir, classifier_dir, ae_dir, ae_type)

    ae.eval()
    backbone.eval()
    classifier.eval()

    files = os.listdir(dataset_dir)
    files = [f for f in files if os.path.isfile(os.path.join(dataset_dir, f))] # list of files in dataset dir.

    random_file = random.choice(files) # Choose a random file

    anchor_id = int(random_file[:4])
    anchor_img_path = os.path.join(dataset_dir, random_file)

    print(f"Selected file: {random_file}")
    print(f"Selected ID: {anchor_id}")

    anchor_img = Image.open(anchor_img_path)
    anchor_img = transform(anchor_img)  # Apply the transformation
    anchor_img = anchor_img.unsqueeze(0)  # Add an extra dimension for the batch size
    anchor_img = anchor_img.cuda()

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
        
        print(f"Processing batch {idx // BATCH_SIZE}")
        images = torch.stack(images)
        images = images.cuda()

        with torch.no_grad():
            backbone_output = backbone(anchor_img)

            if (ae_type == 'vae'):
                    _, anchor_embedding, _, _= ae(backbone_output)
            else:
                _, anchor_embedding = ae(backbone_output)
            
            anchor_embedding = anchor_embedding.repeat(BATCH_SIZE, 1)

            out = backbone(images)

            if (ae_type == 'vae'):
                _, embeddings, _, _ = ae(out)
            else:
                _, embeddings = ae(out)
            
            embeddings = embeddings.view(embeddings.size(0), -1)  # Flatten the embeddings
            concatenated_embeddings = torch.cat((embeddings, anchor_embedding), dim=1)  # Concatenate the embeddings with the anchor embedding
            outputs = classifier(concatenated_embeddings)  # Get the outputs from the classifier
            # outputs = torch.nn.functional.cosine_similarity(embeddings, anchor_embedding, dim=1)  # Get the cosine similarity between the embeddings
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

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

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

    plt.savefig('./res/topk.png')
    plt.show()