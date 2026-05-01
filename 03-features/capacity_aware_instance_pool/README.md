# SageMaker Capacity Aware Instance Pools

This folder contains notebooks demonstrating how to deploy models using **SageMaker Capacity Aware Instance Pools** for real-time inference endpoints.

## Overview

Capacity Aware Instance Pools allow you to define multiple instance types with priority-based fallback, enabling SageMaker to automatically select available capacity and improve deployment reliability.

### Key Features

- **Instance Pools**: Define up to 5 instance types with priority-based fallback
- **Automatic Capacity Management**: SageMaker automatically selects available capacity from your pool
- **LEAST_OUTSTANDING_REQUESTS Routing**: Optimized request distribution across different instance types
- **Autoscaling Support**: Scale seamlessly across different instance types in the pool
- **High Availability**: Fallback to lower-priority instances when preferred capacity is unavailable

## Requirements

- **boto3 >= 1.43.1** (released January 2025)
- **AWS CLI >= 1.45.1** (v1) or **>= 2.34.40** (v2)
- IAM role with SageMaker permissions

## Supported Regions

Capacity Aware Instance Pools are available in **16 AWS regions**:

| Region Code       | Location              | Airport Code |
|-------------------|-----------------------|--------------|
| `ap-south-1`      | Mumbai                | BOM          |
| `us-west-2`       | Portland/Oregon       | PDX          |
| `ca-central-1`    | Montreal              | YUL          |
| `us-east-1`       | N. Virginia           | IAD          |
| `us-east-2`       | Ohio                  | CMH          |
| `ap-northeast-2`  | Seoul                 | ICN          |
| `eu-west-2`       | London                | LHR          |
| `ap-southeast-2`  | Sydney                | SYD          |
| `eu-north-1`      | Stockholm             | ARN          |
| `ap-southeast-3`  | Jakarta               | CGK          |
| `eu-west-1`       | Ireland               | DUB          |
| `eu-central-1`    | Frankfurt             | FRA          |
| `sa-east-1`       | São Paulo             | GRU          |
| `ap-northeast-1`  | Tokyo                 | NRT          |
| `ap-southeast-1`  | Singapore             | SIN          |
| `eu-central-2`    | Zurich                | ZRH          |

## Notebooks

### 1. Single Model Endpoint with Instance Pools
**File**: `single_model_endpoint_with_instance_pools.ipynb`

Deploy a single model on a SageMaker endpoint using capacity aware instance pools.

**What's Included:**
- ✅ Region validation for supported regions
- ✅ boto3 version checking and upgrade instructions
- ✅ Single model deployment with 3 instance types (ml.g6.24xlarge, ml.g5.48xlarge, ml.g5.12xlarge)
- ✅ LEAST_OUTSTANDING_REQUESTS routing configuration
- ✅ Weighted utilization autoscaling policy
- ✅ Comprehensive autoscaling testing methods
- ✅ CloudWatch metrics monitoring
- ✅ Load testing examples
- ✅ Cleanup procedures

**Use Case:**
- Deploy one model with automatic capacity fallback
- Simple deployment without multi-model complexity
- Recommended for most use cases

**Example Instance Pool Configuration:**
```python
"InstancePools": [
    {"InstanceType": "ml.g6.24xlarge", "Priority": 1},  # Try first
    {"InstanceType": "ml.g5.48xlarge", "Priority": 2},  # Fallback
    {"InstanceType": "ml.g5.12xlarge", "Priority": 3},  # Last resort
]
```

### 2. Inference Components with Instance Pools
**File**: `inference_components_with_instance_pools.ipynb`

Deploy multiple models using Inference Components with capacity aware instance pools.

**What's Included:**
- Multiple models on shared infrastructure
- Inference Component configuration
- Resource allocation per model
- Advanced multi-model deployment patterns

**Use Case:**
- Deploy multiple models on the same endpoint
- Share GPU resources across models
- Cost optimization for multiple models

## Quick Start

### Step 1: Choose Your Notebook

- **Single model?** → Use `single_model_endpoint_with_instance_pools.ipynb`
- **Multiple models?** → Use `inference_components_with_instance_pools.ipynb`

### Step 2: Set Your Region

Update the `region` variable in the notebook to one of the 16 supported regions:

```python
region = "us-east-1"  # Change to your preferred region
```

The notebook will validate your region and show an error if it's not supported.

### Step 3: Run the Notebook

1. **Install/upgrade boto3**: The first cell upgrades boto3 to the required version
2. **Restart kernel**: Required after package upgrade
3. **Validate environment**: Checks boto3 version and region support
4. **Configure deployment**: Set model, instance types, and timeout values
5. **Deploy endpoint**: Creates model, endpoint config, and endpoint
6. **Test inference**: Send sample requests to verify deployment
7. **Configure autoscaling** (optional): Set up automatic scaling policies
8. **Test autoscaling** (optional): Validate scaling behavior
9. **Cleanup**: Delete resources to avoid ongoing charges

## Configuration Options

### Instance Pool Configuration

Define up to 5 instance types with priorities:

```python
"InstancePools": [
    {"InstanceType": "ml.g6.24xlarge", "Priority": 1},
    {"InstanceType": "ml.g5.48xlarge", "Priority": 2},
    {"InstanceType": "ml.g5.12xlarge", "Priority": 3},
]
```

**Priority Behavior:**
- SageMaker tries Priority 1 first
- Falls back to Priority 2 if Priority 1 unavailable
- Falls back to Priority 3 if Priority 2 unavailable
- Can mix instance types in the same fleet

### Routing Strategy

**LEAST_OUTSTANDING_REQUESTS (LOR)** is required for capacity aware instance pools:

```python
"RoutingConfig": {"RoutingStrategy": "LEAST_OUTSTANDING_REQUESTS"}
```

**Why LOR?**
- Routes requests to the instance with fewest in-flight requests
- Higher-capacity instances naturally receive more traffic
- No manual weight configuration needed
- Optimal for heterogeneous instance types

### Provision Timeout

Set how long SageMaker should retry capacity procurement:

```python
"VariantInstanceProvisionTimeoutInSeconds": 3600  # 1 hour
```

- SageMaker retries on InsufficientCapacity errors within this window
- Moves to next priority pool once timeout expires
- Separate from model download and container startup timeouts

## Autoscaling

### Weighted Utilization Metric

The notebooks include a weighted utilization autoscaling policy that accounts for different instance capacities:

```python
# Define max concurrency for each instance type
max_concurrency = {
    "ml.g6.24xlarge": 12,
    "ml.g5.48xlarge": 16,
    "ml.g5.12xlarge": 7
}

# Weighted utilization = average of (concurrency / max_capacity) across all types
```

### Scaling Policy Configuration

```python
"TargetValue": 0.7,           # Scale out above 70% utilization
"ScaleInCooldown": 300,       # Wait 5 minutes before scaling in
"ScaleOutCooldown": 60,       # Wait 1 minute before scaling out
"MinCapacity": 1,             # Minimum instances
"MaxCapacity": 10             # Maximum instances
```

### Testing Autoscaling

The `single_model_endpoint_with_instance_pools.ipynb` notebook includes 4 testing methods:

1. **Verify Configuration**: Check scalable target and policy registration
2. **Monitor CloudWatch Metrics**: Query metrics programmatically
3. **Load Testing**: Generate concurrent requests to trigger scale-out
4. **Manual Scaling**: Set desired capacity to verify scaling works

## Best Practices

### Instance Type Selection

1. **Choose similar GPU generations**: Mix G5 and G6 instances, not G4 and G6
2. **Consider throughput ratios**: Balance capacity across instance types
3. **Test with your workload**: Determine max concurrency per instance type
4. **Start with 3 types**: Provides good fallback without over-complication

### Priority Assignment

1. **Priority 1**: Latest generation or best price/performance
2. **Priority 2**: Proven reliability, good availability
3. **Priority 3**: Cost-effective fallback option

### Autoscaling Configuration

1. **Set realistic max concurrency**: Load test to determine actual capacity
2. **Use weighted metrics**: Account for different instance capacities
3. **Tune cooldown periods**: Balance responsiveness vs. stability
4. **Monitor CloudWatch**: Watch for scaling patterns and adjust

### Cost Optimization

1. **Use autoscaling**: Scale down during low traffic periods
2. **Set appropriate MinCapacity**: Balance cost vs. cold start latency
3. **Choose cost-effective fallbacks**: Lower-priority instances can be cheaper
4. **Monitor utilization**: Adjust target value based on actual usage

## Troubleshooting

### InsufficientCapacity Errors

**Symptom**: Endpoint creation fails with capacity errors

**Solutions**:
- Increase `VariantInstanceProvisionTimeoutInSeconds` (default: 3600s)
- Add more instance types to the pool (up to 5 total)
- Try different instance types or regions
- Check if instance types are available in your region

### Autoscaling Not Triggering

**Symptom**: Instance count doesn't change under load

**Solutions**:
- Verify CloudWatch metrics are being published
- Check utilization actually exceeds target value (70%)
- Wait 3-5 minutes for autoscaling to react
- Verify scalable target and policy are registered correctly

### Slow Scaling

**Symptom**: Scaling takes too long to respond

**Solutions**:
- Reduce `ScaleOutCooldown` (currently 60s)
- Lower `TargetValue` threshold (currently 0.7)
- Check instance provision time in CloudWatch

### Aggressive Scaling

**Symptom**: Too many scale-out/scale-in cycles

**Solutions**:
- Increase `TargetValue` threshold
- Increase `ScaleOutCooldown` and `ScaleInCooldown`
- Adjust max concurrency values if they're too conservative

## Documentation

- **API Reference**: [CreateEndpointConfig](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateEndpointConfig.html)
- **Developer Guide**: [Capacity Aware Instance Pools](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-heterogeneous.html)
- **boto3 Changelog**: [v1.43.1 Release](https://github.com/boto/boto3/blob/develop/CHANGELOG.rst)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review CloudWatch logs and metrics
3. Consult AWS SageMaker documentation
4. Contact AWS Support for capacity-related issues

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

These notebooks are provided as sample code for educational and demonstration purposes.

---

**Last Updated**: May 2026  
**Feature Launch**: January 2025 (boto3 1.43.1)
