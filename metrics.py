import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics.utils.metrics import ConfusionMatrix, ap_per_class, box_iou,plot_pr_curve, plot_mc_curve
from pathlib import Path
import cv2
save_dir = Path("results_200")
# Configuration
NUM_CLASSES = 6  # Replace with your actual number of classes
IOU_THRESHOLD = 0.5

GT_LABELS_DIR = "datasets6classyolo/test/labels"
PRED_LABELS_DIR = "final_predictions"
IMAGE_DIR = "datasets6classyolo/test/images"
names ={0: 'people', 1:'bike', 2:'vehicle', 3:'truck', 4:'bus', 5:'motorcycle'}  # Replace with your actual class names
names_tuple = tuple(names.values())
# Helper functions
def load_yolo_labels(path):
    if not os.path.exists(path):
        return np.zeros((0, 5))  # empty file = no labels
    return np.loadtxt(path).reshape(-1, 5)

def load_yolo_predictions(path):
    if not os.path.exists(path):
        return np.zeros((0, 6))
    return np.loadtxt(path).reshape(-1, 6)

def xywhn_to_xyxy(boxes,width,height):
    # YOLO normalized format: x_center, y_center, w, h → x1, y1, x2, y2
    x_c, y_c, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return np.stack([x1*width, y1*height, x2*width, y2*height], axis=1)

# Initialization
confusion_matrix = ConfusionMatrix(nc=NUM_CLASSES)
stats = []

# Processing all images
label_files = sorted(glob.glob(os.path.join(GT_LABELS_DIR, "*.txt")))

for label_path in label_files:
    filename = os.path.basename(label_path)
    image_name = filename.replace(".txt", "")
    gt_data = load_yolo_labels(os.path.join(GT_LABELS_DIR, filename))
    pred_data = load_yolo_predictions(os.path.join(PRED_LABELS_DIR, filename))
    image = cv2.imread(os.path.join(IMAGE_DIR, image_name + ".jpg"))
    width, height = image.shape[1], image.shape[0]
    if gt_data.size == 0:
        gt_classes = torch.tensor([], dtype=torch.int64)
        gt_boxes = torch.zeros((0, 4))
    else:
        gt_classes = torch.tensor(gt_data[:, 0], dtype=torch.int64)
        gt_boxes = torch.tensor(xywhn_to_xyxy(gt_data[:, 1:],width, height))

    if pred_data.size == 0:
        pred_classes = torch.tensor([], dtype=torch.int64)
        pred_boxes = torch.zeros((0, 4))
        pred_scores = torch.tensor([])
    else:
        pred_classes = torch.tensor(pred_data[:, 0], dtype=torch.int64)
        pred_scores = torch.tensor(pred_data[:, 1])
        pred_boxes = torch.tensor(xywhn_to_xyxy(pred_data[:, 2:],width, height))

    # Convert to required format: pred = (n, 6) = x1, y1, x2, y2, conf, cls
    if pred_boxes.shape[0] > 0:
        pred_tensor = torch.cat([pred_boxes, pred_scores.unsqueeze(1), pred_classes.unsqueeze(1)], dim=1)
    else:
        pred_tensor = torch.zeros((0, 6))

    # GT: (m, 5) = cls, x1, y1, x2, y2
    if gt_boxes.shape[0] > 0:
        gt_tensor = torch.cat([gt_classes.unsqueeze(1).float(), gt_boxes], dim=1)
    else:
        gt_tensor = torch.zeros((0, 5))

    # Update confusion matrix
    confusion_matrix.process_batch(pred_tensor, gt_boxes, gt_classes)

    # Collect stats for PR/mAP
    iou_maxes = []
    corrects = []
    if pred_tensor.shape[0]:
        iou = box_iou(pred_boxes, gt_boxes) if gt_boxes.shape[0] else torch.zeros((pred_boxes.shape[0], 0))
        correct = torch.zeros(pred_boxes.shape[0], dtype=torch.bool)
        if iou.shape[1]:
            iou_max, iou_argmax = iou.max(1)
            for i in range(iou.shape[0]):
                if iou_max[i] > IOU_THRESHOLD and pred_classes[i] == gt_classes[iou_argmax[i]]:
                    correct[i] = True
    corrects.append(correct)
    stats.append((correct.unsqueeze(1).detach().cpu().numpy(), pred_scores.detach().cpu().numpy(), pred_classes.detach().cpu().numpy(), gt_classes.detach().cpu().numpy()))

# Compute metrics
corrects = torch.stack(corrects, 0).flatten().detach().cpu().numpy()

if len(stats):
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    tp, fp, p, r, f1, ap, unique_classes, p_curve, r_curve, f1_curve, x, prec_values = ap_per_class(*stats, plot=True, save_dir=save_dir,names=names)
else:
    print("No valid predictions to compute mAP.")
print(f"Accuracy: {corrects.sum()/len(corrects):.4f}")
print(f"mAP@{IOU_THRESHOLD}: {ap.mean():.4f}")
print(pred_tensor[0])
# print(f"Precision: {precision.mean():.4f}")
# print(f"Recall: {recall.mean():.4f}")
# print(f"mAP@0.5: {ap.mean():.4f}")

# Plot confusion matrix
confusion_matrix.plot(save_dir="results_200",names=names_tuple, normalize=True)
confusion_matrix.plot(save_dir="results_200",names=names_tuple, normalize=False)
print("Saved confusion matrix to 'results_200/'")

# Plot PR curve
# if len(ap_class) > 0:
#     fig_pr = plot_pr_curve(ap, ap_class, path="results/pr_curve.png")
#     fig_mc = plot_mc_curve(precision, ap_class, "Precision", "results/precision_curve.png")
#     print("Saved PR and precision curves to 'results/'")
