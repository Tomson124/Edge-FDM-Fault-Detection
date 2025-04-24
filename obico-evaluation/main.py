from lib.detection_model import load_net, detect
from torchvision.datasets import ImageFolder
import cv2
import os
import math
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import torch
from torch.utils.data import random_split

model_cfg = "model/model.cfg"
model_meta = "model/model.meta"
model_weights = "model/model-weights-5a6b1be1fa.onnx"

dataset_name_mapping = {
    'combined_ds': '[1] Prototypical Network Dataset',
    '3D_fault_dataset': '[2] CNN Dataset',
    'reduced_ds': '[3] Prototypical Network Dataset (reduced)',
    'Printing_Dataset': '[4] Beyer, 2024 Dataset (unfiltered)'
}

data_root = os.path.join(os.path.dirname(__file__), '../datasets/')
detection_threshold = 0.25
nms_threshold = 0.0

net_main_1 = load_net(model_cfg, model_meta, model_weights)

eval_datasets = ["reduced_ds", "3D_fault_dataset", "combined_ds", "Printing_Dataset"]

image_transform = transforms.Lambda(lambda img: cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))  # type: ignore

global_total = 0
global_correct = 0
global_loss_sum = 0.0
positive_label = 'failure'
global_TP = global_FP = global_FN = 0
global_time_sum = 0.0

for ds_name in eval_datasets:
    ds_path = os.path.join(data_root, ds_name)
    dataset = ImageFolder(ds_path, transform=image_transform)
    train_ratio = 0.7
    val_ratio = 0.15
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

    ds_start = time.time()
    total = correct = 0
    loss_sum = 0.0
    tp = fp = fn = 0
    eps = 1e-6
    for batch in loader:
        image, label = batch[0]
        total += 1
        dets = detect(net_main_1, image, thresh=detection_threshold, nms=nms_threshold)
        truth = dataset.classes[label]
        scores = [conf for (name, conf, _) in dets if name == truth]
        score_truth = max(scores) if scores else 0.0
        if dets:
            pred, pred_conf, _ = max(dets, key=lambda x: x[1])
        else:
            pred, pred_conf = 'success', 0.0
        if pred == truth:
            correct += 1
        if pred == positive_label and truth == positive_label:
            tp += 1
        elif pred == positive_label and truth != positive_label:
            fp += 1
        elif pred != positive_label and truth == positive_label:
            fn += 1
        loss_sum += -math.log(score_truth + eps)
    ds_end = time.time()
    ds_duration = ds_end - ds_start if (ds_end - ds_start) > 0 else eps
    global_time_sum += ds_duration
    acc = correct / total if total else 0
    avg_loss = loss_sum / total if total else 0
    global_total += total
    global_correct += correct
    global_loss_sum += loss_sum
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
    global_TP += tp
    global_FP += fp
    global_FN += fn
    throughput_ds = total / ds_duration
    print(f"Dataset {ds_name}: Accuracy: {acc:.4f}, Loss: {avg_loss:.4f}, F1: {f1:.4f}, Throughput: {throughput_ds:.2f} img/s")

overall_acc = global_correct / global_total if global_total else 0
overall_loss = global_loss_sum / global_total if global_total else 0
print(f"Overall: Accuracy: {overall_acc:.4f}, Loss: {overall_loss:.4f}")
overall_precision = global_TP / (global_TP + global_FP) if (global_TP + global_FP) else 0
overall_recall = global_TP / (global_TP + global_FN) if (global_TP + global_FN) else 0
overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)) if (overall_precision + overall_recall) else 0
print(f"Overall F1: {overall_f1:.4f}")
overall_throughput = global_total / global_time_sum if global_time_sum else 0
print(f"Overall Throughput: {overall_throughput:.2f} img/s")