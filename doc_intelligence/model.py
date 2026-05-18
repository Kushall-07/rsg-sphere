"""Multi-output neural network for document classification."""

import torch
import torch.nn as nn


class DocumentIntelligenceNet(nn.Module):
    """
    Multi-Output Neural Network.
    Input: 1012 features (1000 TF-IDF + 12 HC)
    Three output heads: Document Type (6), Subject (8), Difficulty (3).
    """

    def __init__(
        self,
        input_dim: int = 1012,
        doc_type_classes: int = 6,
        subject_classes: int = 8,
        difficulty_classes: int = 3,
    ):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.doc_type_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, doc_type_classes),
        )

        self.subject_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, subject_classes),
        )

        self.difficulty_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, difficulty_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        shared = self.shared(x)
        return (
            self.doc_type_head(shared),
            self.subject_head(shared),
            self.difficulty_head(shared),
        )

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
