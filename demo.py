import argparse
from PIL import Image
import torch
import torchvision.transforms as transforms
import cv2
from ultralytics import YOLO

from utils import load_model
import time

import os

BATCH_SIZE = 144
CONF_LEVEL = 0.8

def draw_bbox(img, bbox, text="", color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def demo(args):

    backbone_type = args.backbone_type
    backbone_dir = args.backbone_dir
    classifier_dir = args.classifier_dir
    ae_dir = args.ae_dir
    ae_type = args.ae_type
    gallery_dir = args.gallery_dir
    inst_count = 0
    
    backbone, ae, classifier = load_model(backbone_type, backbone_dir, classifier_dir, ae_dir, ae_type)

    backbone = backbone.cuda()
    ae = ae.cuda()
    classifier = classifier.cuda()

    backbone.eval()
    ae.eval()
    classifier.eval()

    yolo = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225)),
    ])

    cap = cv2.VideoCapture('./demo/demo.mp4')    
    if (not cap.isOpened()): 
        print("Error opening video stream or file")
 
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            results = yolo.predict(source=frame, save=False, classes=[0], conf=0.7)
            results = results[0]

            boxes = results.boxes.xyxy

            inst_count = 0  # Ensure this is initialized outside the loop

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                # Crop the person from the original image
                person_image = frame[y1:y2, x1:x2]
                person_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
                person_image = Image.fromarray(person_image)
                person_image_tensor = transform(person_image)
                person_image_tensor = person_image_tensor.unsqueeze(0).cuda()

                query_found = False

                with torch.no_grad():
                    feat = backbone(person_image_tensor)

                    if (ae_type == 'vae'):
                        _, embed, _, _= ae(feat)
                    else:
                        _, embed = ae(feat)
                    
                    query_found = False
                    id = -1

                    gal_imgs = []
                    gal_lbs = []

                    for filename in os.listdir(gallery_dir):
                        if filename.endswith('.jpg') or filename.endswith('.png'):
                            img_path = os.path.join(gallery_dir, filename)
                            img = Image.open(img_path)
                            img_t = transform(img)
                            img_t = img_t.cuda()
                            gal_imgs.append(img_t)
                            gal_lbs.append(int(filename[:4]))

                        print(f"Processing image {len(gal_imgs)}")
                        if len(gal_imgs) >= BATCH_SIZE:
                            print("Processing a full batch")

                            gal_imgs = torch.stack(gal_imgs).cuda()
                            gal_lbs = torch.tensor(gal_lbs).cuda()

                            embed = embed.repeat(gal_imgs.size(0), 1) 
                            gal_feats = backbone(gal_imgs) 
                            if ae_type == 'vae':
                                _, gal_embeds, _, _ = ae(gal_feats)
                            else:
                                _, gal_embeds = ae(gal_feats)

                            concat = torch.cat((embed, gal_embeds), dim=1)
                            output = classifier(concat) 
                            print(output)
                            mask = output >= CONF_LEVEL
                            indices = torch.nonzero(mask).flatten()

                            if indices.numel() > 0:
                                query_found = True
                                id = gal_lbs[indices[0]].item()
                                break

                            # Reset gal_imgs and gal_lbs for the next batch
                            gal_imgs = []
                            gal_lbs = []

                    # Handle the case when no query was found after all batches
                    if not query_found:
                        inst_count += 1
                        print(f"New person: {inst_count} detected")
                        id = inst_count
                        # Save person image in gallery_dir, ensure person_image and frame are defined
                        person_image.save(os.path.join(gallery_dir, f'{id:04d}_c1s1_000000.jpg'))
                        
                draw_bbox(frame, (x1, y1, x2, y2), text=f'ID: {id}')
                    
                # save person image
                # person_image.save(f'./demo/person_{idx}.jpg')
                # person_image = Image.fromarray(person_image)
                
                # draw_bbox(frame, (x1, y1, x2, y2), text=results.names[idx])
                    
                
            """people_boxes = results.xyxy[0][results.xyxy[0][:, 5] == 0]  # class '0' for person

            for i, bbox in enumerate(people_boxes):
                # Convert bounding box coordinates from float to integer
                xmin, ymin, xmax, ymax = map(int, bbox[:4])
                
                # Crop the person from the original image
                person_image = frame.crop((xmin, ymin, xmax, ymax))
                
                # Save or display the image
                person_image.save(f'person_{i}.jpg')"""
                            
            
            cv2.imshow('Frame', frame)
        
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        else: 
            break
        
    # When everything done, release the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone_dir', type=str, default="/home/bilginer/Downloads/best_backbone.pkl", help='backbone weights directory')
    parser.add_argument('--backbone_type', type=str, default="dense", help='backbone name')
    parser.add_argument('--classifier_dir', type=str, default="/home/bilginer/Downloads/best_classifier.pkl", help='Autoencoder weights')
    parser.add_argument('--ae_dir', type=str, default="/home/bilginer/Downloads/best_ae.pkl", help='autoencoder weights')
    parser.add_argument('--ae_type', type=str, default="ae", help='autoencoder type')
    parser.add_argument('--gallery_dir', type=str, default=None, help='gallery directory')

    args = parser.parse_args()
    demo(args)