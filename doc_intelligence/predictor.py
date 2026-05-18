"""Inference for uploaded PDF documents."""

from __future__ import annotations

import torch

from doc_intelligence.feature_extractor import DocumentFeatureExtractor
from doc_intelligence.model import DocumentIntelligenceNet


class DocumentPredictor:
    DOC_TYPE_LABELS = [
        ("📚 Textbook", "#F5C518"),
        ("📝 Question Paper", "#E64833"),
        ("📋 Notes", "#90AEAD"),
        ("🔬 Lab Manual", "#874F41"),
        ("🔭 Research Paper", "#7C3AED"),
        ("📅 Syllabus", "#2563EB"),
    ]

    SUBJECT_LABELS = [
        ("💻 Computer Science", "#F5C518"),
        ("📐 Mathematics", "#E64833"),
        ("⚛️ Physics", "#90AEAD"),
        ("⚡ Electronics", "#874F41"),
        ("🧪 Chemistry", "#7C3AED"),
        ("⚙️ Mechanical", "#2563EB"),
        ("🏗️ Civil", "#00b09b"),
        ("📰 General", "#FF6B6B"),
    ]

    DIFFICULTY_LABELS = [
        ("🟢 Beginner", "#00b09b"),
        ("🟡 Intermediate", "#F5C518"),
        ("🔴 Advanced", "#E64833"),
    ]

    def __init__(self):
        self.model = None
        self.extractor = None
        self.is_loaded = False

    def load(
        self,
        model_path: str = "models/doc_intelligence.pth",
        extractor_path: str = "models/feature_extractor.pkl",
    ) -> bool:
        try:
            try:
                checkpoint = torch.load(
                    model_path, map_location="cpu", weights_only=False
                )
            except TypeError:
                checkpoint = torch.load(model_path, map_location="cpu")
            input_dim = checkpoint.get("input_dim", 1012)
            self.model = DocumentIntelligenceNet(input_dim=input_dim)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()

            self.extractor = DocumentFeatureExtractor()
            self.extractor.load(extractor_path)

            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Model load error: {e}")
            return False

    def predict_pdf(self, pdf_file):
        if not self.is_loaded:
            return None

        features, text = self.extractor.extract_from_pdf(pdf_file)
        X = torch.FloatTensor(features)

        with torch.no_grad():
            out_type, out_sub, out_diff = self.model(X)

        prob_type = torch.softmax(out_type, dim=1)[0].tolist()
        prob_sub = torch.softmax(out_sub, dim=1)[0].tolist()
        prob_diff = torch.softmax(out_diff, dim=1)[0].tolist()

        type_idx = int(max(range(len(prob_type)), key=lambda i: prob_type[i]))
        sub_idx = int(max(range(len(prob_sub)), key=lambda i: prob_sub[i]))
        diff_idx = int(max(range(len(prob_diff)), key=lambda i: prob_diff[i]))

        return {
            "doc_type": {
                "label": self.DOC_TYPE_LABELS[type_idx][0],
                "color": self.DOC_TYPE_LABELS[type_idx][1],
                "confidence": max(prob_type) * 100,
                "all_probs": [
                    (self.DOC_TYPE_LABELS[i][0], p * 100)
                    for i, p in enumerate(prob_type)
                ],
            },
            "subject": {
                "label": self.SUBJECT_LABELS[sub_idx][0],
                "color": self.SUBJECT_LABELS[sub_idx][1],
                "confidence": max(prob_sub) * 100,
                "all_probs": [
                    (self.SUBJECT_LABELS[i][0], p * 100)
                    for i, p in enumerate(prob_sub)
                ],
            },
            "difficulty": {
                "label": self.DIFFICULTY_LABELS[diff_idx][0],
                "color": self.DIFFICULTY_LABELS[diff_idx][1],
                "confidence": max(prob_diff) * 100,
                "all_probs": [
                    (self.DIFFICULTY_LABELS[i][0], p * 100)
                    for i, p in enumerate(prob_diff)
                ],
            },
            "text_preview": text[:200],
        }
