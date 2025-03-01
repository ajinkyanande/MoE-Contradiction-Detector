# **MoE-Contradiction-Detector**

## **Overview**
MoE-Contradiction-Detector is an AI-powered tool designed to detect contradictions and ensure factual consistency between two text sources. It leverages a **Mixture of Experts (MoE) model** to classify entailments and contradictions in sentence pairs. This tool is particularly useful for **fact-checking, automated text validation, and NLP research**.

## **Architecture**
The project is structured into the following key components:

1. **Model Training** – Uses **PyTorch Lightning** to train the MoE model on the **SNLI dataset**.
2. **Model Optimization** – Supports **quantization, pruning, and ONNX conversion** for improved efficiency.
3. **Inference** – Provides both **PyTorch- and ONNX-based** inference pipelines.
4. **Backend** – Implements a **FastAPI** server to handle text comparison requests.
5. **Frontend** – A **Streamlit** web app for user interaction and visualization.
6. **Docker Integration** – Includes scripts for **building, running, and deploying** Docker containers.

---

## **Model Details**
The **MoE model** consists of:
- A **Transformer encoder** to process input sentence pairs.
- A **Mixture of Experts (MoE) layer** to classify relationships (entailment, contradiction, or neutral).

The model is trained on the **Stanford Natural Language Inference (SNLI) dataset**.

---

## **Installation & Setup**
### **1. Setting Up the Environment**
Install the required **Conda** environment:

```sh
conda env create -f environment.yml
```

Activate the environment:

```sh
conda activate moe-cd
```

### **2. Training & Testing**
Ensure `config.yaml` is configured with the required hyperparameters.

#### **Training**
```sh
python -m src.train
```

#### **Testing**
```sh
python -m src.inference [-h] {torch,onnx}
```

---

## **Model Optimization**
Enhance model efficiency with **quantization, pruning, and ONNX conversion**:

### **Quantize the Model**
```sh
python -m src.optimize quantize <ckpt_path> <output_path>
```

### **Prune the Model**
```sh
python -m src.optimize prune <ckpt_path> <output_path> --amount=0.2
```

### **Convert to ONNX Format**
```sh
python -m src.optimize onnx <ckpt_path> <onnx_path>
```

---

## **Local Deployment**
### **1. Run the Backend (FastAPI)**
```sh
uvicorn backend.app:app --host 0.0.0.0 --port 8080
```

### **2. Run the Frontend (Streamlit)**
```sh
streamlit run frontend/app.py
```

---

## **Deployment Using Docker**
### **1. Build Docker Image**
```sh
sudo sh scripts/docker_build.sh --name=moe-contradiction-detector --tag=latest
```

### **2. Run Docker Container Locally**
```sh
sudo sh scripts/docker_run_local.sh --name=moe-contradiction-detector --tag=latest --port=8080
```

### **3. Push Docker Image to AWS ECR**
```sh
sudo -E sh scripts/docker_push_ecr.sh --name=moe-contradiction-detector --tag=latest --region=<aws_region> --account=<aws_account_id>
```
