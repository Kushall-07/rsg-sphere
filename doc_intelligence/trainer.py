"""Training loop for the Document Intelligence neural network."""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from doc_intelligence.model import DocumentIntelligenceNet


class DocumentModelTrainer:
    def __init__(self):
        self.model = DocumentIntelligenceNet()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.001, weight_decay=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.5
        )
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "doc_type_acc": [],
            "subject_acc": [],
            "difficulty_acc": [],
            "lr": [],
        }

    def _compute_loss(self, out_type, out_sub, out_diff, y_type, y_sub, y_diff):
        return (
            0.4 * self.criterion(out_type, y_type)
            + 0.4 * self.criterion(out_sub, y_sub)
            + 0.2 * self.criterion(out_diff, y_diff)
        )

    def _accuracy(self, outputs, labels):
        preds = outputs.argmax(dim=1)
        return (preds == labels).float().mean().item()

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        for X, y_type, y_sub, y_diff in train_loader:
            self.optimizer.zero_grad()
            out_type, out_sub, out_diff = self.model(X)
            loss = self._compute_loss(out_type, out_sub, out_diff, y_type, y_sub, y_diff)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        acc_type = acc_sub = acc_diff = 0.0
        with torch.no_grad():
            for X, y_type, y_sub, y_diff in val_loader:
                out_type, out_sub, out_diff = self.model(X)
                loss = self._compute_loss(out_type, out_sub, out_diff, y_type, y_sub, y_diff)
                total_loss += loss.item()
                acc_type += self._accuracy(out_type, y_type)
                acc_sub += self._accuracy(out_sub, y_sub)
                acc_diff += self._accuracy(out_diff, y_diff)
        n = len(val_loader)
        return {
            "val_loss": total_loss / n,
            "doc_type_acc": acc_type / n * 100,
            "subject_acc": acc_sub / n * 100,
            "difficulty_acc": acc_diff / n * 100,
        }

    def train_full(self, features, labels, epochs: int = 30, batch_size: int = 32):
        X = torch.FloatTensor(features)
        y_type = torch.LongTensor(labels["doc_type_ids"])
        y_sub = torch.LongTensor(labels["subject_ids"])
        y_diff = torch.LongTensor(labels["difficulty_ids"])

        indices = list(range(len(X)))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=0.2,
            random_state=42,
            stratify=labels["doc_type_ids"],
        )

        train_ds = TensorDataset(
            X[train_idx], y_type[train_idx], y_sub[train_idx], y_diff[train_idx]
        )
        val_ds = TensorDataset(
            X[val_idx], y_type[val_idx], y_sub[val_idx], y_diff[val_idx]
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, drop_last=False
        )

        print(f"Training: {len(train_idx)} | Validation: {len(val_idx)}")
        print(f"Starting {epochs} epoch training...\n")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            self.scheduler.step()

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["doc_type_acc"].append(val_metrics["doc_type_acc"])
            self.history["subject_acc"].append(val_metrics["subject_acc"])
            self.history["difficulty_acc"].append(val_metrics["difficulty_acc"])
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            print(
                f"Epoch {epoch:2d}/{epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"DocType: {val_metrics['doc_type_acc']:.1f}% | "
                f"Subject: {val_metrics['subject_acc']:.1f}% | "
                f"Diff: {val_metrics['difficulty_acc']:.1f}%"
            )

            if epoch % 10 == 0:
                self.save_checkpoint(epoch)

        print("\nTraining Complete!")
        print(f"Final DocType Acc: {self.history['doc_type_acc'][-1]:.1f}%")
        print(f"Final Subject Acc: {self.history['subject_acc'][-1]:.1f}%")
        print(f"Final Difficulty Acc: {self.history['difficulty_acc'][-1]:.1f}%")

        return self.history

    def save_checkpoint(self, epoch: int):
        os.makedirs("models", exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "history": self.history,
            },
            f"models/checkpoint_e{epoch}.pth",
        )

    def save_final(self):
        os.makedirs("models", exist_ok=True)

        torch.save(
            {
                "model_state": self.model.state_dict(),
                "history": self.history,
                "input_dim": 1012,
                "doc_type_classes": 6,
                "subject_classes": 8,
                "difficulty_classes": 3,
            },
            "models/doc_intelligence.pth",
        )

        with open("models/training_history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f)

        final_metrics = {
            "doc_type_acc": self.history["doc_type_acc"][-1],
            "subject_acc": self.history["subject_acc"][-1],
            "difficulty_acc": self.history["difficulty_acc"][-1],
            "final_loss": self.history["val_loss"][-1],
            "total_epochs": len(self.history["epoch"]),
            "total_params": self.model.get_param_count(),
        }
        with open("models/final_metrics.json", "w", encoding="utf-8") as f:
            json.dump(final_metrics, f)

        print("Model saved to models/doc_intelligence.pth")
        print("History saved to models/training_history.json")
        print("Metrics saved to models/final_metrics.json")
