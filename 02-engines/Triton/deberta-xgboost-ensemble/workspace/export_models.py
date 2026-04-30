"""
Step 1: Prepare Models

Exports DeBERTa-v3-base model and trains/exports XGBoost classifier to ONNX format.

Both models are exported to ONNX for use with Triton's onnxruntime backend.

Outputs:
- workspace/deberta/: DeBERTa ONNX model with mean pooling
- workspace/xgboost/: XGBoost ONNX model

Usage:
    python workspace/export_models.py [--workspace WORKSPACE] [--max-seq-len 128]
    python workspace/export_models.py --step deberta   # export DeBERTa only
    python workspace/export_models.py --step xgboost   # train XGBoost only
"""

import argparse
import os


def export_deberta_to_onnx(workspace: str, max_seq_len: int = 128):
    """Export DeBERTa-v3-base model to ONNX format with mean pooling."""
    import torch
    import torch.nn as nn
    from transformers import DebertaV2Model

    model_id = "microsoft/deberta-v3-base"
    dest = os.path.join(workspace, "deberta")
    os.makedirs(dest, exist_ok=True)

    print("  Downloading DeBERTa-v3-base model...")
    base_model = DebertaV2Model.from_pretrained(model_id)
    base_model.eval()

    # Create wrapper with mean pooling
    class DeBERTaWithPooling(nn.Module):
        """DeBERTa model with mean pooling for ONNX export."""

        def __init__(self, base_model):
            super().__init__()
            self.deberta = base_model

        def forward(self, input_ids, attention_mask):
            outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state = outputs.last_hidden_state  # (batch, seq_len, 768)

            # Mean pooling: average hidden states weighted by attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
            sum_hidden = torch.sum(hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_hidden / sum_mask  # (batch, 768)

            return embeddings

    # Wrap model with pooling
    wrapped_model = DeBERTaWithPooling(base_model)
    wrapped_model.eval()

    print(f"  Exporting to ONNX (max_seq_len={max_seq_len})...")

    # Create dummy inputs for export
    dummy_input_ids = torch.zeros((1, max_seq_len), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, max_seq_len), dtype=torch.long)

    # Export to ONNX
    onnx_path = os.path.join(dest, "model.onnx")
    torch.onnx.export(
        wrapped_model,
        (dummy_input_ids, dummy_attention_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "embeddings": {0: "batch"},
        },
        opset_version=18,
        do_constant_folding=True,
    )

    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  ✓ DeBERTa ONNX model saved to {dest}")
    print(f"    - model.onnx ({onnx_size_mb:.1f} MB)")
    print(f"    - Includes mean pooling layer")
    print(f"    - Input: input_ids, attention_mask (max_len={max_seq_len})")
    print(f"    - Output: embeddings (768-dim)")


def train_and_export_xgboost(workspace: str):
    """Train XGBoost classifier and export to ONNX format."""
    import xgboost as xgb
    from sklearn.datasets import make_classification
    import onnx
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    dest = os.path.join(workspace, "xgboost")
    os.makedirs(dest, exist_ok=True)

    # Generate synthetic 768-dimensional training data (matching DeBERTa embeddings)
    print("  Generating synthetic 768-dim training data...")
    X_train, y_train = make_classification(
        n_samples=1000,
        n_features=768,
        n_informative=50,
        n_redundant=10,
        n_classes=2,
        random_state=42,
    )

    # Train XGBoost binary classifier
    print("  Training XGBoost classifier...")
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective="binary:logistic",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Convert to ONNX format
    print("  Converting XGBoost model to ONNX...")
    initial_type = [("float_input", FloatTensorType([None, 768]))]
    onnx_model = convert_xgboost(clf, initial_types=initial_type, target_opset=12)

    # Save ONNX model
    onnx_path = os.path.join(dest, "model.onnx")
    onnx.save_model(onnx_model, onnx_path)

    # Validate ONNX model outputs
    print("  Validating ONNX model outputs...")
    loaded_model = onnx.load(onnx_path)
    output_names = [output.name for output in loaded_model.graph.output]
    expected_outputs = ["label", "probabilities"]

    if set(output_names) != set(expected_outputs):
        print(f"  ⚠ WARNING: ONNX outputs {output_names} don't match expected {expected_outputs}")
        print(f"  Update config.py if this changes in future onnxmltools versions")
    else:
        print(f"  ✓ ONNX outputs validated: {output_names}")

    print(f"  ✓ XGBoost ONNX model saved to {dest}")
    print(f"    - model.onnx ({os.path.getsize(onnx_path) / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Export DeBERTa and train XGBoost to ONNX"
    )
    parser.add_argument(
        "--workspace",
        default="workspace",
        help="Local workspace directory for model artifacts (default: workspace)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="Maximum sequence length for DeBERTa tokenization (default: 128)",
    )
    parser.add_argument(
        "--step",
        choices=["deberta", "xgboost"],
        default=None,
        help="Run a single step instead of both (avoids process state conflicts)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Step 1: Prepare Models (ONNX)")
    print("=" * 70)
    print()

    # Create workspace directory
    os.makedirs(args.workspace, exist_ok=True)

    if args.step is None or args.step == "deberta":
        print("[1/2] Exporting DeBERTa-v3-base to ONNX...")
        export_deberta_to_onnx(args.workspace, args.max_seq_len)
        print()

    if args.step is None or args.step == "xgboost":
        print("[2/2] Training and exporting XGBoost to ONNX...")
        train_and_export_xgboost(args.workspace)
        print()

    print("=" * 70)
    print("✓ Step 1 Complete")
    print("=" * 70)
    print()
    print(f"Models saved to: {os.path.abspath(args.workspace)}")
    print("  - deberta/model.onnx (DeBERTa with mean pooling)")
    print("  - xgboost/model.onnx (XGBoost classifier)")
    print()
    print("Both models use ONNX format - no Python dependencies needed in Triton!")


if __name__ == "__main__":
    main()
