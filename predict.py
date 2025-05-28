#%%
from ultralytics import YOLO
import os
from utils import *
from geomloss import SamplesLoss
from torchvision import transforms
import torch
import joblib
if __name__ == "__main__":
    #%%
    RGB = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    K= 100
    class_prototypes_path = f'class_prototypes_{K}.pt'
    class_prototypes_labels_path = f'class_prototypes_labels_{K}.pt'
    dist_matrix_path = f'dist_matrix_{K}.pt'
    knn_classifier_path = f'knn_classifier_{K}.joblib'
    results_path = f'results_{K}.pt'
    # print("Loading YOLOv11 model")
    # model = YOLO('yolo11m_1class/weights/best.pt').eval().to(device)
    # print("Loading YOLOv11 model done")
    # print("Detecting objects")
    # results1 = model.predict('datasets6classyolo/test/images', verbose=False)
    # print("Detecting objects done")
    # print("Saving annotations")

    dest_folder1 = 'predicted_annotations'
    # if not os.path.exists(dest_folder1):
    #     os.makedirs(dest_folder1)
    # extract_annotations(results1,dest_folder1)
    # print("Annotations saved")
    ##############classification##############
    test_dir = 'objects/test'
    detected_objects_dir = 'detected_objects'
    #load resnet
    #%%
    if RGB:
        resnet = None
    else:
        print("Loading resnet50 model")
        resnet_path = 'best_resnet_32.pth'
        num_classes = 6
        full_resnet = get_resnet50(resnet_path, num_classes)
        resnet = get_feature_extractor(full_resnet)
        resnet = resnet.to(device).eval()
        
        print("Loading resnet50 model done")
    #%%
    print("Loading train dataset")
    train_dir = 'objects/train'
    transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    train_dataset, train_loader = get_train_dataset_loader(train_dir,transform, batch_size=1)
    print("Loading train dataset done")
    class_names = train_dataset.classes
    print(f"class_names: {class_names}")
    #%%
    # get class prototypes and labels
    prot_exists = os.path.exists(class_prototypes_path)
    prot_labels_exists = os.path.exists(class_prototypes_labels_path)
    if prot_exists:
        print("Loading class prototypes")
        class_prototypes = torch.load(class_prototypes_path, map_location=device)
    if prot_labels_exists:
        print("Loading class prototypes labels")
        class_prototypes_labels = torch.load(class_prototypes_labels_path, map_location=device)
    if not(prot_exists and prot_labels_exists):
        print("Building class prototypes and labels")
        class_prototypes, class_prototypes_labels = get_class_prototypes_and_labels(train_loader,class_names, resnet,K=K, device=device)
        print("Saving class prototypes and labels")
        torch.save(class_prototypes, class_prototypes_path)
        torch.save(class_prototypes_labels, class_prototypes_labels_path)
        print(f"Class prototypes and labels saved as {class_prototypes_path} and {class_prototypes_labels_path}")
    class_prototypes_labels = torch.stack(class_prototypes_labels, dim=0).to(device)
    class_prototypes_labels = class_prototypes_labels.view(-1)
    class_prototypes_labels = class_prototypes_labels.detach().cpu().numpy()
    print(f"len(class_prototypes): {len(class_prototypes)}")
    print(f"shape: {class_prototypes[0].shape}")
    print(f"labels shape: {class_prototypes_labels.shape}")
    #%%
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05, backend="tensorized",debias=False).to(device)

    # get knn classifier
    knn_classifier_exists = os.path.exists(knn_classifier_path)
    if knn_classifier_exists:
        print("Loading knn classifier")
        knn_classifier = joblib.load(knn_classifier_path)
    else:
        
        dist_matrix_exists = os.path.exists(dist_matrix_path)
        if dist_matrix_exists:
            print("Loading distance matrix")
            dist_matrix = torch.load(dist_matrix_path, map_location=device)
            dist_matrix = dist_matrix.to(device)
            print("Distance matrix loaded")
        else:
            print("Computing distance matrix")
            dist_matrix = get_dist_matrix(class_prototypes, loss,device=device)
            dist_matrix = nn.functional.relu(dist_matrix)
            print("Saving distance matrix")
            torch.save(dist_matrix, dist_matrix_path)
            print(f"Distance matrix saved as {dist_matrix_path}")
        dist_matrix = dist_matrix.detach().cpu().numpy()
        # dist_matrix = dist_matrix * 1e5
        print("Building knn classifier")
        knn_classifier = get_knn_classifier(dist_matrix, class_prototypes_labels,n_neighbors=10)
        print("Saving knn classifier")
        joblib.dump(knn_classifier, knn_classifier_path)
        print(f"Knn classifier saved as {knn_classifier_path}")
        
    # classify images
    #%%
    results2_exists = os.path.exists(results_path)
    if results2_exists:
        print("Loading classification results")
        results2 = torch.load(results_path, map_location=device)
        results2 = results2.to(device)
    else:
        print("Classifying images")
        results2 = classify(detected_objects_dir, modelknn, transform,device=device,extract_features=extract_features,
                        feature_extractor=resnet,knn_classifier=knn_classifier, class_prototypes=class_prototypes,
                        loss=loss)
        print("Saving classification results")
        torch.save(results2, results_path)
    dest_folder2 = f'final_predictions_{K}'
    if not os.path.exists(dest_folder2):
        os.makedirs(dest_folder2)
    print("Saving final predictions")
    predict_annotations(results2,dest_folder2,dest_folder1)
    print(f"Final predictions saved in {dest_folder2}")
    print("Done")


    
