# Deploy Black Forest Labs FLUX Models on Amazon SageMaker

This folder contains examples for deploying [Black Forest Labs](https://blackforestlabs.ai/) FLUX image generation models on Amazon SageMaker AI endpoints.

## Examples

### [FLUX.2-dev](./FLUX.2-dev/)

Deploy [FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev), a 32 billion parameter rectified flow transformer for text-to-image generation, single-reference editing, and multi-reference editing — all without finetuning.

| Setting | Value |
|---|---|
| Container | vLLM-Omni (`vllm:omni-sagemaker-cuda-v1`) |
| Instance | ml.g7e.12xlarge (2 GPUs) |
| Tensor Parallel | 2 |
| Model | `black-forest-labs/FLUX.2-dev` |
| API Route | `/v1/images/generations` (via `CustomAttributes` header) |

The notebook uses the SageMaker vLLM-Omni inference container, which serves omni-modality models through OpenAI-compatible APIs. Configuration is handled via `SM_VLLM_*` environment variables and requires `al2023-ami-sagemaker-inference-gpu-4-1` for CUDA 12.9 compatibility.

Key features of FLUX.2 [dev]:
- State-of-the-art open text-to-image generation and editing
- Character, object, and style reference without additional training
- Guidance distillation for efficient inference
- Built-in safety filters (NSFW/CSAM) and content provenance (watermarking, C2PA)

> **License:** [FLUX Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/LICENSE.md)

---

### [Flux.1-Kontext-dev (Optimised)](./Flux.1-Kontext-dev/)

Deploy a [Pruna](https://github.com/PrunaAI)-optimised version of [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) for image editing, achieving a **3.5x speed-up** (63s → 21s) with similar output quality.

| Setting | Value |
|---|---|
| Container | HuggingFace Inference (PyTorch 2.6.0 / Transformers 4.49.0 / Python 3.12) |
| Instance | ml.g6e.xlarge (model preparation and inference) |
| Model | `black-forest-labs/FLUX.1-Kontext-dev` |
| Optimisation | Pruna (FoRA caching, TorchAO int8dq quantisation, torch.compile) |

This example follows a two-step workflow:

1. **Model preparation** — A SageMaker Training Job downloads the model, applies Pruna optimisations (FoRA caching, int8 dynamic quantisation, torch.compile), runs a warmup inference to build the compilation graph, and saves the smashed model to S3.
2. **Deployment** — The optimised model artifact is deployed to a SageMaker real-time endpoint using the HuggingFace Inference container with a custom `inference.py` script.

The inference endpoint accepts a base64-encoded input image and a text prompt, returning edited images as base64-encoded JPEG.

> **License:** [FLUX.1-dev Non-Commercial License](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev)

## Getting Started

1. Open the desired notebook in SageMaker Studio or a Jupyter environment with appropriate AWS permissions.
2. Set your Hugging Face token (`HF_TOKEN`) where indicated — both models are gated on Hugging Face.
3. Ensure your execution role has permissions to create SageMaker endpoints, training jobs, and pull ECR images.
4. Run the cells sequentially.
5. Remember to run the cleanup cell when done to avoid ongoing charges.
