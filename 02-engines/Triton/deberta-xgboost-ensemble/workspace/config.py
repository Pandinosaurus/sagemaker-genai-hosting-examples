"""
Configuration and constants for Triton DeBERTa + XGBoost ensemble deployment.

Contains:
- Triton model configuration generators (config.pbtxt files)
- Benchmark sample texts

Both models use ONNX format with Triton's onnxruntime backend.
"""


# ---------------------------------------------------------------------------
# Triton Model Configuration Generators
# ---------------------------------------------------------------------------

def get_deberta_config(max_seq_len: int = 128) -> str:
    """
    Generate DeBERTa Triton config with specified sequence length.

    Args:
        max_seq_len: Maximum sequence length (must match ONNX export)

    Returns:
        Triton config.pbtxt content as string
    """
    return f"""
name: "deberta"
backend: "onnxruntime"
max_batch_size: 32

input [
  {{
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ {max_seq_len} ]
  }},
  {{
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ {max_seq_len} ]
  }}
]

output [
  {{
    name: "embeddings"
    data_type: TYPE_FP32
    dims: [ 768 ]
  }}
]

instance_group [
  {{ kind: KIND_GPU, count: 1 }}
]
""".strip()


def get_xgb_config() -> str:
    """
    Generate XGBoost Triton config.

    Returns:
        Triton config.pbtxt content as string
    """
    return """
name: "xgboost_classifier"
backend: "onnxruntime"
max_batch_size: 32

input [
  {
    name: "float_input"
    data_type: TYPE_FP32
    dims: [ 768 ]
  }
]

output [
  {
    name: "label"
    data_type: TYPE_INT64
    dims: [ 1 ]
    reshape { shape: [ ] }
  },
  {
    name: "probabilities"
    data_type: TYPE_FP32
    dims: [ 2 ]
  }
]

instance_group [
  { kind: KIND_CPU, count: 1 }
]
""".strip()


def get_ensemble_config(max_seq_len: int = 128) -> str:
    """
    Generate ensemble Triton config with specified sequence length.

    Args:
        max_seq_len: Maximum sequence length (must match ONNX export)

    Returns:
        Triton config.pbtxt content as string
    """
    return f"""
name: "ensemble_model"
platform: "ensemble"
max_batch_size: 32

input [
  {{
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ {max_seq_len} ]
  }},
  {{
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ {max_seq_len} ]
  }}
]

output [
  {{
    name: "PREDICTION"
    data_type: TYPE_INT64
    dims: [ 1 ]
  }}
]

ensemble_scheduling {{
  step [
    {{
      model_name: "deberta"
      model_version: -1
      input_map {{
        key: "input_ids"
        value: "input_ids"
      }}
      input_map {{
        key: "attention_mask"
        value: "attention_mask"
      }}
      output_map {{
        key: "embeddings"
        value: "embeddings"
      }}
    }},
    {{
      model_name: "xgboost_classifier"
      model_version: -1
      input_map {{
        key: "float_input"
        value: "embeddings"
      }}
      output_map {{
        key: "label"
        value: "PREDICTION"
      }}
    }}
  ]
}}
""".strip()


# ---------------------------------------------------------------------------
# Benchmark Texts
# ---------------------------------------------------------------------------

BENCHMARK_TEXTS = [
    "This is a fantastic product! Highly recommend.",
    "Terrible experience, would not buy again.",
    "Average quality, nothing special.",
    "Absolutely love it, will purchase again.",
    "Not worth the price at all.",
    "Decent item, met my expectations.",
    "Best purchase I've made this year!",
    "Very disappointed with the quality.",
]
