import os
import numpy as np
import cv2 
import torch
from torchvision import transforms,models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from geomloss import SamplesLoss
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
from tqdm import tqdm

def extract_annotations(results,dest_folder):
    for result in results:
        img_path=result.path
        img_file=img_path.split('/')[-1]
        img_basename=img_file.split('.')[0]
        annotation_file=f"{img_basename}.txt"
        annotation_path=os.path.join(dest_folder,annotation_file)
        file=open(annotation_path,'w')
        annotations=[]
        boxes=result.boxes
        cls=boxes.cls
        n=cls.size(0)
        conf=boxes.conf
        xywhn=boxes.xywhn
        for i in range(n):
            annotations.append(f"{int(cls[i])} {xywhn[i,0]} {xywhn[i,1]} {xywhn[i,2]} {xywhn[i,3]} {conf[i]}")
        file.write('\n'.join(annotations))
        file.close()
        
def load_yolo_labels(path):
    if not os.path.exists(path):
        return np.zeros((0, 5))  # empty file = no labels
    return np.loadtxt(path).reshape(-1, 5)

def load_yolo_predictions(path):
    if not os.path.exists(path):
        return np.zeros((0, 6))
    return np.loadtxt(path).reshape(-1, 6)

def xywhn_to_xyxy(boxes):
    # YOLO normalized format: x_center, y_center, w, h → x1, y1, x2, y2
    x_c, y_c, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)

def extract_obj(img_path, label_path, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the image
    image = cv2.imread(img_path)
    h, w, _ = image.shape

    # Read the label file
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) == 5:
            class_id, x_center, y_center, box_w, box_h = map(float, parts)
            class_id = int(class_id)
        else:
            class_id, x_center, y_center, box_w, box_h, conf = map(float, parts)
            class_id = int(class_id)

        # Convert YOLO to pixel coordinates
        x_center *= w
        y_center *= h
        box_w *= w
        box_h *= h

        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)

        # Ensure bounding box is within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            print(f"⚠️  Skipping invalid box in {label_path}, line {idx}")
            continue
        
        # Crop the image
        object_crop = image[y1:y2, x1:x2]
        if object_crop.size == 0:
            print(f"⚠️  Skipping empty crop in {label_path}, line {idx}")
            continue
        # Save the cropped object
        output_path = os.path.join(output_dir, f"{os.path.basename(img_path).split('.')[0]}__{idx}.jpg")
        cv2.imwrite(output_path, object_crop)
        
def classify(objects_dir, model, transform,device='cuda',**kwargs):
    results = {}
    for img_name in tqdm(os.listdir(objects_dir)):
        obj_basename = img_name.split('.')[0]
        img_basename = obj_basename.split('__')[0]
        obj_idx = int(obj_basename.split('__')[1])
        if img_basename not in results:
            results[img_basename] = []
        img_path = os.path.join(objects_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image,**kwargs)
            
            results[img_basename].append((obj_idx, output.item()))
    return results

def get_resnet50(resnet_path, num_classes):
    resnet = models.resnet50(pretrained=False)
    resnet.fc = nn.Sequential(
        nn.Linear(resnet.fc.in_features, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes)
    )
    resnet.load_state_dict(torch.load(resnet_path))
    return resnet

def modelknn(image_tensor, extract_features,feature_extractor, knn_classifier, class_prototypes,loss,device='cuda'):
    batch_size = image_tensor.size(0)
    image_tensor = image_tensor.to(device)
    K = class_prototypes[0].size(0)
    num_classes = len(class_prototypes)
    img_features = extract_features(feature_extractor,image_tensor)
    if feature_extractor is not None:
        img_features = img_features.unsqueeze(-1)
    img_features = torch.repeat_interleave(img_features,K, dim=0)
    dists = []
    for c in range(num_classes):
        class_prototype = class_prototypes[c].to(device)
        feature_dim = class_prototype.size(1)
        if feature_extractor is not None:
            class_prototype = class_prototype.unsqueeze(0)
            class_prototype = class_prototype.unsqueeze(-1)
            class_prototype = class_prototype.expand(batch_size, K, feature_dim, 1)
            class_prototype = class_prototype.contiguous().view(batch_size*K, feature_dim, 1)
        else:
            class_prototype = class_prototype.unsqueeze(0)
            class_prototype = class_prototype.expand(batch_size, K, feature_dim,3)
            class_prototype = class_prototype.contiguous().view(batch_size*K, feature_dim, 3)
        L = loss(img_features, class_prototype)
        L = L.view(batch_size, K)
        dists.append(L)
    dists = torch.stack(dists, dim=1)
    dists = dists.view(batch_size, -1)
    dists = dists.detach().cpu().numpy()
    dists = np.maximum(dists, 0)
    preds = knn_classifier.predict(dists)
    return preds

def get_feature_extractor(resnet, device='cuda'):
    feature_extractor = nn.Sequential(
        *list(resnet.children())[:-1],  # everything before .fc
        nn.Flatten(),
        *list(resnet.fc[:-1])           # remove classification head
    )
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor

def extract_features(feature_extractor, input,device='cuda'):
    input = input.to(device)
    if feature_extractor is not None:
        with torch.no_grad():
            features = feature_extractor(input)
            features = features.squeeze(-1)
        # features = nn.functional.relu(features)
        # features = features/ (features.sum(dim=1).unsqueeze(0).transpose(1,0))
        # features = nn.functional.softmax(features, dim=1)
    else:
        features = input.permute(0, 2, 3, 1)  # NCHW to NHWC
        features = features.view(features.size(0), -1, 3)  # NHWC to NWHC
    return features.to(device)
def predict_annotations(results,dest_folder,annotation_folder):
    class_dict = {0:1, 1:4, 2:5, 3:0, 4:3, 5:2}
    for basename in results:
        f_orig=open(os.path.join(annotation_folder,basename+'.txt'),'r')
        f_pred=open(os.path.join(dest_folder,basename+'.txt'),'w')
        lines=f_orig.readlines()
        for obj_idx, pred in results[basename]:
            line=lines[obj_idx]
            parts=line.strip().split()
            class_id, x_center, y_center, box_w, box_h, conf = map(float, parts)
            f_pred.write(f"{class_dict[pred]} {conf} {x_center} {y_center} {box_w} {box_h} \n")
        f_orig.close()
        f_pred.close()

def get_train_dataset_loader(train_dir,transform, batch_size=1, num_workers=4):
    dataset = ImageFolder(root=train_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataset,train_loader

def get_class_prototypes_and_labels(train_loader,class_names, resnet,K=100, device='cuda'):
    num_classes = len(class_names)
    class_prototypes = [[] for _ in range(num_classes)]
    class_prototypes_labels = [[] for _ in range(num_classes)]
    for image, label in train_loader:
        image = image[0].to(device)
        label = label[0].item()
        if len(class_prototypes[label]) < K:
            class_prototypes[label].append(image)
            class_prototypes_labels[label].append(label)
        elif all([len(class_prototypes[i]) == K for i in range(num_classes)]):
            break
    for i in range(num_classes):
        class_prototypes[i] = torch.stack(class_prototypes[i], dim=0).to(device)
        class_prototypes_labels[i] = torch.tensor(class_prototypes_labels[i]).to(device)
        class_prototypes[i] = extract_features(resnet, class_prototypes[i], device=device)
        if resnet is not None:
            class_prototypes[i] = class_prototypes[i].squeeze(-1)
    return class_prototypes, class_prototypes_labels

def get_knn_classifier(dist_matrix, class_prototypes_labels, n_neighbors=20):
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric='precomputed')
    knn_classifier.fit(dist_matrix, class_prototypes_labels)
    return knn_classifier

def get_dist_matrix(class_prototypes, loss, device='cuda'):
    num_classes = len(class_prototypes)  # = 6
    K           = class_prototypes[0].size(0)  # = 100
    feature_dim = class_prototypes[0].size(1)  # = 2048
    channel_dim = class_prototypes[0].size(2)  # = 3 (RGB)
    dist_matrix = torch.zeros((num_classes*K, num_classes*K), device=device)
    
    # First, build a flat list of all 600 “(Ni, 1)-shaped” prototypes:
    flat_protos = []
    for c in range(num_classes):
        # class_prototypes[c] is (100, 2048). We want (100, 2048, 1)
        proto = class_prototypes[c].to(device).view(K, feature_dim, channel_dim)
        for i in range(K):
            flat_protos.append(proto[i : i+1])  
            # each proto[i:i+1] is shape (1, 2048, 1)
    
    # Now flat_protos has length 600, each is (1, 2048, 1). 
    # We compute distances pairwise (600×600):
    for i in tqdm(range(num_classes*K)):
        x_i = flat_protos[i]  # (1, 2048, 1)
        for j in range(i+1, num_classes*K):
            y_j = flat_protos[j]  # (1, 2048, 1)
            with torch.no_grad():
                d_ij = loss(x_i, y_j)  # cost‐matrix (1, 2048, 2048) ~ 16 MB
            dist_matrix[i, j] = d_ij
            dist_matrix[j, i] = d_ij
    
    return dist_matrix

