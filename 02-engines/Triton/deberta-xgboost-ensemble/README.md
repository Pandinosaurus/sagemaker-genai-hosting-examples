# DeBERTa + XGBoost Ensemble on Triton

Deploy a two-stage model ensemble to Amazon SageMaker on NVIDIA Triton:

- **DeBERTa-v3-base** (ONNX) — text encoder producing 768-d mean-pooled embeddings
- **XGBoost classifier** (ONNX) — downstream classifier operating on the embeddings

Both models run on Triton's ONNX Runtime backend and are chained server-side via a Triton ensemble, so the client issues a single request and the embedding tensor never leaves the GPU host.

## Why a Triton ensemble

Running encoder + classifier as separate endpoints doubles network hops and forces the 768-d embedding tensor across the wire. With a Triton ensemble, the embedding stays in Triton's shared memory between the two model invocations — one client request, one GPU host.

## Files

```
deberta-xgboost-ensemble/
├── deberta_xgboost_triton_ensemble.ipynb   # End-to-end deployment notebook
└── workspace/
    ├── export_models.py                    # DeBERTa ONNX export + XGBoost train/export
    ├── build_triton_repo.py                # Assemble Triton model repository
    ├── config.py                           # Triton config.pbtxt generators
    └── run_benchmark.py                    # Latency/throughput benchmark
```

Start with the notebook.
