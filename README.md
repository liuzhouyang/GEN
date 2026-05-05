# Rethinking Flexible Graph Similarity Computation: One-step Alignment with Global Guidance

This repository contains the official implementation of the ICDE 2026 accepted paper:

**"Rethinking Flexible Graph Similarity Computation: One-step Alignment with Global Guidance"**

## Installation

Please ensure the following dependencies are installed:

- Python 3.9+
- PyTorch
- PyTorch Geometric
- NetworkX
- NumPy
- CUDA for GPU acceleration

## Running the Code

To run GEN on a specific dataset, use the following command:

```
python -m run --dataset_name <dataset_name>
```

For example, to run on the AIDS dataset:

```
python -m run --dataset_name aids
```

If no dataset name is specified, the AIDS dataset will be used by default.

To run baselines, use the following command:

```
python -m benchmarking.<baseline_name>.run --dataset_name <dataset_name>
```

