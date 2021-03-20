import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from utils import top_k_acc as _top_k_acc


def top_1_acc(output, target):
    return _top_k_acc(output, target, k=1)


def top_3_acc(output, target):
    return _top_k_acc(output, target, k=3)


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def AUC(output, target):
    output = np.array(output)
    target = np.array(target)
    scores = []
    for i in range(target.shape[1]):
        score = roc_auc_score(target[:, i], output[:, i])
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score

