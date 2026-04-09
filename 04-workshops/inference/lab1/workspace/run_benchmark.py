"""
Step 4: Benchmark Endpoint

Runs latency and throughput benchmarks against a deployed Triton endpoint.
Supports Inference Component routing via --inference-component-name.

Note: The endpoint expects tokenized inputs (input_ids, attention_mask), so this script
performs client-side tokenization using the DeBERTa tokenizer.

Inputs:
- Deployed SageMaker endpoint

Outputs:
- Latency statistics (min, mean, median, p90, p95, p99, max)
- Throughput metrics (requests per second)
- Sample predictions

Usage:
    python workspace/run_benchmark.py --endpoint-name <name> --inference-component-name <name> [--warmup 5] [--iterations 50] [--concurrency 1]
"""

import argparse
import json
import statistics
import time
import concurrent.futures

import boto3
import numpy as np
from transformers import DebertaV2Tokenizer

from config import BENCHMARK_TEXTS

# Load tokenizer globally for reuse
_tokenizer = None


def get_tokenizer(max_seq_len: int = 128):
    """Get or create the DeBERTa tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        print("Loading DeBERTa tokenizer...")
        _tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
        _tokenizer.model_max_length = max_seq_len
        print("✓ Tokenizer loaded")
    return _tokenizer


def tokenize_text(text: str, max_seq_len: int = 128):
    """Tokenize text for DeBERTa model."""
    tokenizer = get_tokenizer(max_seq_len)
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="np",
    )
    return encoded["input_ids"][0].tolist(), encoded["attention_mask"][0].tolist()


def invoke_endpoint(
    runtime,
    endpoint_name: str,
    text: str,
    max_seq_len: int = 128,
    inference_component_name: str | None = None,
) -> tuple[int, float]:
    """
    Invoke the Triton endpoint with a single text input.

    Returns:
        tuple: (prediction, latency_ms)
    """
    input_ids, attention_mask = tokenize_text(text, max_seq_len)

    payload = {
        "inputs": [
            {
                "name": "input_ids",
                "shape": [1, max_seq_len],
                "datatype": "INT64",
                "data": [input_ids],
            },
            {
                "name": "attention_mask",
                "shape": [1, max_seq_len],
                "datatype": "INT64",
                "data": [attention_mask],
            },
        ]
    }

    invoke_kwargs = dict(
        EndpointName=endpoint_name,
        ContentType="application/octet-stream",
        Body=json.dumps(payload),
    )
    if inference_component_name:
        invoke_kwargs["InferenceComponentName"] = inference_component_name

    t0 = time.perf_counter()
    response = runtime.invoke_endpoint(**invoke_kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    result = json.loads(response["Body"].read().decode("utf-8"))
    prediction = result["outputs"][0]["data"][0]

    return int(prediction), latency_ms


def _worker(args):
    """Worker function for concurrent benchmarking."""
    runtime, endpoint_name, text, max_seq_len, ic_name = args
    return invoke_endpoint(runtime, endpoint_name, text, max_seq_len, ic_name)


def run_benchmark(
    endpoint_name: str,
    region: str,
    warmup: int = 5,
    iterations: int = 50,
    concurrency: int = 1,
    max_seq_len: int = 128,
    inference_component_name: str | None = None,
):
    """Run benchmark against the deployed endpoint."""

    runtime = boto3.client("sagemaker-runtime", region_name=region)
    texts = BENCHMARK_TEXTS

    def cycle_text(i):
        return texts[i % len(texts)]

    print("=" * 70)
    print("Benchmark Configuration")
    print("=" * 70)
    print(f"  Endpoint: {endpoint_name}")
    if inference_component_name:
        print(f"  Inference Component: {inference_component_name}")
    print(f"  Region: {region}")
    print(f"  Max sequence length: {max_seq_len}")
    print(f"  Warmup requests: {warmup}")
    print(f"  Benchmark requests: {iterations}")
    print(f"  Concurrency: {concurrency}")
    print()

    # Load tokenizer (will print message on first load)
    get_tokenizer(max_seq_len)
    print()

    # Warmup phase
    print(f"Running {warmup} warmup request(s)...")
    for i in range(warmup):
        invoke_endpoint(runtime, endpoint_name, cycle_text(i), max_seq_len, inference_component_name)
    print("✓ Warmup complete")
    print()

    # Benchmark phase
    print(f"Running {iterations} benchmark request(s)...")
    latencies = []
    errors = 0

    if concurrency == 1:
        # Sequential execution
        for i in range(iterations):
            try:
                _, ms = invoke_endpoint(runtime, endpoint_name, cycle_text(i), max_seq_len, inference_component_name)
                latencies.append(ms)
            except Exception as e:
                errors += 1
                print(f"  [ERROR] Request {i}: {e}")
    else:
        # Concurrent execution
        work = [
            (runtime, endpoint_name, cycle_text(i), max_seq_len, inference_component_name)
            for i in range(iterations)
        ]
        wall_start = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_worker, w): idx for idx, w in enumerate(work)}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    _, ms = fut.result()
                    latencies.append(ms)
                except Exception as e:
                    errors += 1
                    print(f"  [ERROR]: {e}")

        wall_ms = (time.perf_counter() - wall_start) * 1000.0

    print("✓ Benchmark complete")
    print()

    # Compute statistics
    if not latencies:
        print("No successful requests - cannot compute stats")
        return

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    def percentile(p):
        idx = max(0, int(np.ceil(p / 100.0 * n)) - 1)
        return latencies_sorted[idx]

    print("=" * 70)
    print("Benchmark Results")
    print("=" * 70)
    print(f"  Successful requests: {n}")
    print(f"  Failed requests: {errors}")
    print()
    print("Latency Statistics:")
    print(f"  Min      : {min(latencies):.1f} ms")
    print(f"  Mean     : {statistics.mean(latencies):.1f} ms")
    print(f"  Median   : {statistics.median(latencies):.1f} ms")
    print(f"  p90      : {percentile(90):.1f} ms")
    print(f"  p95      : {percentile(95):.1f} ms")
    print(f"  p99      : {percentile(99):.1f} ms")
    print(f"  Max      : {max(latencies):.1f} ms")

    if n > 1:
        print(f"  StdDev   : {statistics.stdev(latencies):.1f} ms")

    if concurrency > 1:
        rps = n / (wall_ms / 1000.0)
        print()
        print("Throughput:")
        print(f"  Requests/sec: {rps:.1f} req/s  (concurrency={concurrency})")

    # Show sample predictions
    print()
    print("Sample Predictions:")
    for text in BENCHMARK_TEXTS[:4]:
        pred, ms = invoke_endpoint(runtime, endpoint_name, text, max_seq_len, inference_component_name)
        print(f"  [{pred}]  ({ms:.0f} ms)  \"{text[:55]}\"")

    print()


def main():
    parser = argparse.ArgumentParser(description="Step 4: Benchmark Triton endpoint")
    parser.add_argument(
        "--endpoint-name",
        required=True,
        help="SageMaker endpoint name to benchmark",
    )
    parser.add_argument(
        "--inference-component-name",
        default=None,
        help="Inference component name (required for IC-based deployments)",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region (default: from boto3 session)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization (default: 128)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup requests (default: 5)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of benchmark requests (default: 50)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent threads (default: 1)",
    )
    args = parser.parse_args()

    # Determine region
    region = args.region
    if not region:
        session = boto3.Session()
        region = session.region_name

    # Run benchmark
    run_benchmark(
        endpoint_name=args.endpoint_name,
        region=region,
        warmup=args.warmup,
        iterations=args.iterations,
        concurrency=args.concurrency,
        max_seq_len=args.max_seq_len,
        inference_component_name=args.inference_component_name,
    )

    print("=" * 70)
    print("✓ Step 4 Complete")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
