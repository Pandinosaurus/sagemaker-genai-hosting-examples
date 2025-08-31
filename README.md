# Generative AI Inference Examples on Amazon SageMaker
This repository contains a compilation of examples of optimized deployment of popular Large Language Models (LLMs) utilizing SageMaker Inference. Hosting LLMs comes with a variety of challenges due to the size of the model, inefficient usage of hardware, and scaling LLMs into a production like environment with multiple concurrent users.

SageMaker Inference is a highly performant and versatile hosting option that comes with a variety of options that you can utilize to efficiently host your LLMs. In this repository we showcase how you can take different SageMaker Inference options such as Real-Time Inference (low latency, high throughput use-cases) and Asynchronous Inference (near real-time/batch use-cases) and integrate with Model Servers such as [DJL Serving](https://github.com/deepjavalibrary/djl-serving) and [Text Generation Inference](https://github.com/huggingface/text-generation-inference). We showcase how you can tune for performance via optimizing these different Model Serving stacks and also exploring hardware options such as [Inferentia2](https://aws.amazon.com/blogs/machine-learning/achieve-high-performance-with-lowest-cost-for-generative-ai-inference-using-aws-inferentia2-and-aws-trainium-on-amazon-sagemaker/) integration with Amazon SageMaker.

## Content
If you are contributing, please add a link to your model below:

- [Mistral](./Mistral/)
- [Mixtral](./Mistral/)
- [Falcon](./Falcon/)
- [Flan](./FlanT5/)
- [Llama2](./Llama2/)
- [Llama3](./Llama3/)
- [Open-Llama](./Open-Llama/)
- [Zephyr](./Zephyr/)
- [CodeGen](./Codegen25/)
- [CodeLlama](./CodeLlama/)

## Additional Resources

- [Introduction to Large Model Inference Container](https://aws.amazon.com/blogs/machine-learning/boost-inference-performance-for-llms-with-new-amazon-sagemaker-containers/)
- [LLM Inference Optimization Toolkit](https://aws.amazon.com/blogs/machine-learning/achieve-up-to-2x-higher-throughput-while-reducing-costs-by-50-for-generative-ai-inference-on-amazon-sagemaker-with-the-new-inference-optimization-toolkit-part-1/)
- [Large Model Inference Container Tuning Guide](https://docs.djl.ai/docs/serving/serving/docs/lmi/tuning_guides/deepspeed_tuning_guide.html)
- [Text Generation Inference with Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/announcing-the-launch-of-new-hugging-face-llm-inference-containers-on-amazon-sagemaker/)
- [Server Side Batching Optimizations with LMI](https://aws.amazon.com/blogs/machine-learning/improve-throughput-performance-of-llama-2-models-using-amazon-sagemaker/)
- [General SageMaker Hosting Examples Repo](https://github.com/aws-samples/sagemaker-hosting)
- [SageMaker Hosting Blog Series](https://aws.amazon.com/blogs/machine-learning/model-hosting-patterns-in-amazon-sagemaker-part-1-common-design-patterns-for-building-ml-applications-on-amazon-sagemaker/)
- [Easily deploy and manage hundreds of LoRA Adapters](https://aws.amazon.com/blogs/machine-learning/easily-deploy-and-manage-hundreds-of-lora-adapters-with-sagemaker-efficient-multi-adapter-inference/)

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](./LICENSE) file.

