# Deploy FLUX.2-Dev on Amazon SageMaker using the vLLM-Omni Container

This example demonstrates how to deploy [FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) by Black Forest Labs to an Amazon SageMaker real-time endpoint using the **vLLM-Omni** inference container and the AWS Python SDK (boto3).

---

## FLUX.2 [dev]

**FLUX.2 [dev]** is a **32 billion parameter rectified flow transformer** by [Black Forest Labs](https://blackforestlabs.ai/) capable of generating, editing, and combining images based on text instructions.

### Key Features

- **State-of-the-art** open text-to-image generation, single-reference editing, and multi-reference editing
- **No finetuning required** — supports character, object, and style reference without additional training, all in one model
- **Guidance distillation** — trained with guidance distillation for more efficient inference
- **Open weights** — released to drive scientific research and empower artists to develop innovative workflows

### Safety

The model includes pre-training data filtering (NSFW/CSAM), post-training safety fine-tuning against adversarial attacks, and built-in NSFW/IP-infringing content filters at input and output. Content provenance features (pixel-layer watermarking, C2PA metadata) are also supported.

### Availability

FLUX.2 [dev] is supported in [Diffusers](https://github.com/huggingface/diffusers), [ComfyUI](https://github.com/comfyanonymous/ComfyUI), and Black Forest Labs' own [reference implementation](https://github.com/black-forest-labs/flux).

> **License:** [FLUX Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/LICENSE.md)

---

## vLLM-Omni Inference Container

**vLLM-Omni** is a pre-built Docker container for serving **omni-modality models** — including text-to-speech, image generation, video generation, and multimodal chat — through **OpenAI-compatible APIs**. It is built on Amazon Linux 2023 with CUDA 12.9 and Python 3.12.

### Supported Modalities

| Modality | Route | Example Model |
|---|---|---|
| Text-to-Speech | `/v1/audio/speech` | Qwen3-TTS-12Hz-1.7B-CustomVoice |
| Image Generation | `/v1/images/generations` | FLUX.2-klein-4B |
| Video Generation | `/v1/videos` | Wan2.1-T2V-1.3B-Diffusers |
| Multimodal Chat | `/v1/chat/completions` | Qwen2.5-Omni-3B, BAGEL-7B-MoT |

### SageMaker Integration

A dedicated SageMaker image (`vllm:omni-sagemaker-cuda-v1`) is available. Configuration is done via `SM_VLLM_*` environment variables (e.g., `SM_VLLM_MODEL`, `SM_VLLM_MAX_MODEL_LEN`, `SM_VLLM_TENSOR_PARALLEL_SIZE`), which are automatically converted to vLLM CLI arguments. GPU deployments require `inference_ami_version="al2023-ami-sagemaker-inference-gpu-4-1"` for CUDA 12.9 compatibility.

### Routing Middleware

The SageMaker image includes an **ASGI routing middleware** that dispatches the standard `/invocations` endpoint to the correct vLLM-Omni route based on the `CustomAttributes` header (e.g., `CustomAttributes="route=/v1/images/generations"`). Without a route, requests fall through to the default vLLM `/invocations` handler.

> **Documentation:** [vLLM-Omni Inference — AWS Deep Learning Containers](https://aws.github.io/deep-learning-containers/vllm-omni/)

---

## Deployment Configuration

| Setting | Value |
|---|---|
| Container Image | `vllm:omni-sagemaker-cuda-v1` |
| Instance Type | `ml.g7e.12xlarge` (2 GPUs) |
| Tensor Parallel Size | 2 |
| Model ID | `black-forest-labs/FLUX.2-dev` |
| API Route | `/v1/images/generations` |
| Startup Health Check Timeout | 900 seconds |
| Inference AMI Version | `al2023-ami-sagemaker-inference-gpu-4-1` |

### Environment Variables

| Variable | Description |
|---|---|
| `HF_TOKEN` | Your Hugging Face access token (model is gated) |
| `SM_NUM_GPUS` | Number of GPUs on the instance |
| `SM_VLLM_MODEL` | HuggingFace model ID (`black-forest-labs/FLUX.2-dev`) |
| `SM_VLLM_TENSOR_PARALLEL_SIZE` | Number of GPUs for tensor parallelism |

---

## Notebook Walkthrough

The [FLUX.2-dev.ipynb](./FLUX.2-dev.ipynb) notebook covers the following steps:

1. **Setup** — Install/upgrade `boto3` and initialize SageMaker clients.
2. **IAM Role** — Retrieve the SageMaker execution role (or set your own).
3. **Container Configuration** — Define the vLLM-Omni container image URI and environment variables.
4. **Model Creation** — Register the model with SageMaker using `create_model`.
5. **Endpoint Deployment** — Create an endpoint configuration and deploy the endpoint on `ml.g7e.12xlarge`.
6. **Inference** — Send image generation requests via `invoke_endpoint` using the OpenAI-compatible `/v1/images/generations` route, with the route specified through the `CustomAttributes` header.
7. **Cleanup** — Delete the endpoint, endpoint configuration, and model to avoid ongoing charges.

---

## Getting Started

### Prerequisites

- An AWS account with permissions to create SageMaker endpoints and pull ECR images.
- A [Hugging Face](https://huggingface.co/) account with access to the gated [FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) model.
- SageMaker Studio or a Jupyter environment configured with appropriate AWS credentials.

### Steps

1. Open [FLUX.2-dev.ipynb](./FLUX.2-dev.ipynb) in SageMaker Studio or your Jupyter environment.
2. Replace `<YOUR_TOKEN>` with your Hugging Face access token.
3. Run the cells sequentially.
4. **Remember to run the cleanup cell when done** to delete the endpoint and avoid ongoing charges.
