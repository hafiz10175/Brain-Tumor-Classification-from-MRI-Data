

import numpy as np
import torch
import torch.nn as nn
from collections import Counter


def compute_class_weights(labels, num_classes: int):

    counts = Counter(labels)
    weights = []
    for c in range(num_classes):
        n_c = counts.get(c, 1)
        w_c = 1.0 / np.log(1.0 + n_c)
        weights.append(w_c)
    return torch.tensor(weights, dtype=torch.float32)


class LabelSmoothingCrossEntropy(nn.Module):


    def __init__(self, class_weights, epsilon: float = 0.05, num_classes: int = 4):
        super().__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits, target):

        log_probs = torch.log_softmax(logits, dim=-1)  # (N, C)
        with torch.no_grad():
            # one-hot with smoothing
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.epsilon / self.num_classes)
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.epsilon)

        # apply class weights
        # weight for each class: (C,)
        weights = self.class_weights.unsqueeze(0)  # (1, C)
        loss = -(true_dist * log_probs * weights).sum(dim=1).mean()
        return loss


class EarlyStopping:


    def __init__(self, patience: int = 8, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            return

        improve = (score > self.best_score) if self.mode == "max" else (score < self.best_score)

        if improve:
            self.best_score = score
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.should_stop = True
