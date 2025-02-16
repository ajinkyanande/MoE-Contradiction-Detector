# MoE-Contradiction-Detector

## Overview

MoE-Contradiction-Detector is an AI-powered tool designed to detect contradictions and ensure factual consistency between two text sources. It leverages a Mixture of Experts (MoE) model to classify entailments and contradictions in sentence pairs.

## Architecture

The project consists of several key components:

1. **Model Training**: Uses PyTorch Lightning to train the MoE model on the SNLI dataset.
2. **Model Optimization**: Includes scripts for quantizing, pruning, and converting the model to ONNX format.
3. **Inference**: Provides both PyTorch and ONNX inference capabilities.
4. **Backend**: A FastAPI server that handles text comparison requests and runs inference.
5. **Frontend**: A Streamlit application for user interaction and visualization of results.
6. **Docker**: Scripts for building, running, and pushing Docker images.

## Model

The MoE model consists of a Transformer encoder and a Mixture of Experts layer. The Transformer encoder processes the input sentence pairs, while the MoE layer classifies the relationship between them. The model is trained on the SNLI dataset.

## Deployment

### Docker Deployment

#### Build Docker Image

```sh
sudo sh scripts/docker_build.sh --name=moe-contradiction-detector --tag=latest
```

#### Run Docker Container Locally

```sh
sudo sh scripts/docker_run_local.sh --name=moe-contradiction-detector --tag=latest --port=8080
```

#### Push Docker Image to AWS ECR

```sh
sudo -E sh scripts/docker_push_ecr.sh --name=moe-contradiction-detector --tag=latest --region=<aws_region> --account=<aws_account_id>
```

### Local Deployment

#### Backend

1. Install dependencies:

```sh
pip install -r requirements.txt
```

2. Run the FastAPI server:

```sh
uvicorn backend.app:app --host 0.0.0.0 --port 8080
```

#### Frontend

1. Install dependencies:

```sh
pip install streamlit
```

2. Run the Streamlit app:

```sh
streamlit run frontend/app.py
```

## Commands

### Training the Model

```sh
python src/train.py
```

### Optimizing the Model

#### Quantize Model

```sh
python src/optimize.py quantize <ckpt_path> <output_path>
```

#### Prune Model

```sh
python src/optimize.py prune <ckpt_path> <output_path> --amount=0.2
```

#### Convert to ONNX

```sh
python src/optimize.py onnx <ckpt_path> <onnx_path>
```

### Inference

#### PyTorch Inference

```sh
python src/inference.py
```

#### ONNX Inference

```sh
python src/inference.py --onnx
```

## Contact

Developed by: **Ajinkya Nande**

- [LinkedIn](https://www.linkedin.com/in/ajinkyanande/)
- [GitHub](https://github.com/ajinkyanande)
- [Project Repository](https://github.com/ajinkyanande/MoE-Contradiction-Detector)
```
