# Deploy DeepSeek-V4 on Amazon SageMaker AI

This example demonstrates how to deploy **DeepSeek-V4** series models on an Amazon SageMaker AI real-time endpoint.

## Models

DeepSeek-V4 is a family of Mixture-of-Experts (MoE) language models supporting a context length of **one million tokens**:

| Model | Parameters | Activated Parameters |
| :--- | :--- | :--- |
| [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | 1.6T | 49B |
| [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) | 284B | 13B |

### Key Architecture Highlights

- **Hybrid Attention Architecture** — Compressed Sparse Attention (CSA) + Heavily Compressed Attention (HCA) for long-context efficiency. In the 1M-token setting, DeepSeek-V4-Pro requires only 27% of single-token inference FLOPs and 10% of KV cache compared with DeepSeek-V3.2.
- **Manifold-Constrained Hyper-Connections (mHC)** — Strengthens residual connections for improved signal propagation and training stability.
- **Muon Optimizer** — Faster convergence and greater training stability.
- **Native Multi-Token Prediction (MTP)** — Enables speculative decoding for higher throughput with no quality degradation.

## Serving Frameworks

The notebook provides deployment configurations for two serving options:

1. **vLLM (0.20.1)** — Uses the SageMaker DLC with latency-optimized, balanced, and throughput-optimized configurations.
2. **SGLang (BYOC)** — Requires a custom-built container. Refer to the [SGLang DeepSeek-V4 documentation](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4) for details.

## Instance Requirements

- **Instance type:** `ml.p5en.48xlarge` (8x NVIDIA H200 GPUs)

## Notebook Contents

| Section | Description |
| :--- | :--- |
| Container Setup | Configure vLLM or SGLang serving environment |
| Deployment | Create SageMaker Model, Endpoint Configuration, and Endpoint |
| Text Inference | Basic text generation |
| Reasoning | Inference with thinking/reasoning enabled and configurable effort |
| Streaming | Streaming response with token-per-second metrics |
| Cleanup | Delete endpoint and associated resources |

## Getting Started

1. Open the notebook in SageMaker Studio or a Jupyter environment with AWS credentials.
2. Install dependencies (`boto3`).
3. Set your Hugging Face token (`HF_TOKEN`) in the environment variables cell.
4. Choose a serving framework (vLLM or SGLang) and run the corresponding cell.
5. Run the deployment cells and wait for the endpoint to become `InService`.
6. Run inference examples.
7. Clean up resources when done.

## Prerequisites

- AWS account with access to `ml.p5en.48xlarge` instances (capacity reservation recommended for availability).
- IAM role with SageMaker execution permissions.
- Hugging Face access token with permission to download DeepSeek-V4 model weights.
