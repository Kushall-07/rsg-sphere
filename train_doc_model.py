"""
RAG-Sphere — Document Intelligence Model
Private Training Script

Run this once to train and save the model.
Usage: python train_doc_model.py
"""

import os


def main():
    print("=" * 50)
    print("RAG-Sphere Document Intelligence")
    print("Neural Network Training Script")
    print("=" * 50)

    print("\n[1/4] Building dataset...")
    from doc_intelligence.dataset_builder import build_dataset

    df = build_dataset()
    print(f"Dataset: {len(df)} samples")
    print(df["doc_type"].value_counts())

    print("\n[2/4] Extracting features...")
    from doc_intelligence.feature_extractor import DocumentFeatureExtractor

    extractor = DocumentFeatureExtractor(max_features=1000)
    features = extractor.fit_transform(df["text"].tolist())
    print(f"Features shape: {features.shape}")

    os.makedirs("models", exist_ok=True)
    extractor.save("models/feature_extractor.pkl")
    print("Feature extractor saved")

    print("\n[3/4] Training Neural Network...")
    print("Epochs: 30 | Batch: 32 | LR: 0.001")
    print("-" * 50)

    from doc_intelligence.trainer import DocumentModelTrainer

    labels = {
        "doc_type_ids": df["doc_type_id"].tolist(),
        "subject_ids": df["subject_id"].tolist(),
        "difficulty_ids": df["difficulty_id"].tolist(),
    }

    trainer = DocumentModelTrainer()
    history = trainer.train_full(features, labels, epochs=30, batch_size=32)

    print("\n[4/4] Saving model...")
    trainer.save_final()

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print(f"DocType Accuracy:  {history['doc_type_acc'][-1]:.1f}%")
    print(f"Subject Accuracy:  {history['subject_acc'][-1]:.1f}%")
    print(f"Difficulty Accuracy: {history['difficulty_acc'][-1]:.1f}%")
    print(f"Final Loss:        {history['val_loss'][-1]:.4f}")
    print("=" * 50)
    print("\nFiles saved:")
    print("  models/doc_intelligence.pth")
    print("  models/feature_extractor.pkl")
    print("  models/training_history.json")
    print("  models/final_metrics.json")
    print("\nNow run: streamlit run app.py")


if __name__ == "__main__":
    main()
