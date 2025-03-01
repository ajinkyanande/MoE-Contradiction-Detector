# **MoE-Contradiction-Detector**

## **Overview**
MoE-Contradiction-Detector is an **AI-powered tool** designed to detect contradictions and ensure factual consistency between two text sources. It leverages a **Mixture of Experts (MoE) model** to classify entailments and contradictions in sentence pairs. This tool is particularly useful for **fact-checking, automated text validation, and NLP research**.

## **Architecture**
The project is structured into the following key components:

1. **Two-Step Inference Approach**
   - The backend first computes **text similarity using embeddings** to determine related sentence pairs.
   - If the similarity score exceeds a set threshold, the **MoE-based contradiction detection model** evaluates whether the sentences entail or contradict each other.

2. **Tech Stack**
   - **Backend:** FastAPI (handles text processing and model inference).
   - **Frontend:** Streamlit (provides an interactive UI).
   - **Model:** PyTorch & Hugging Face (MoE architecture using MiniLM (BERT-based transformer) experts).
   - **Model Training:** PyTorch Lightning (fine-tuned on SNLI dataset with LoRA).

3. **Model Optimization**
   - Supports **quantization, pruning, and ONNX conversion**.

4. **Inference Pipelines**
   - Provides both **PyTorch- and ONNX-based** inference pipelines.

5. **Deployment**
   - **Dockerized** for local and cloud deployment.

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

---

## **Checkpoints**

Download the following model checkpoints and update the `config.yaml` file with the correct paths:

- **PyTorch Model**: https://drive.google.com/file/d/14Jsz5BgldJuA3tv1rYBWk9lGnmtNKXxa/view?usp=sharing
- **ONNX Model**: https://drive.google.com/file/d/1MGP244rBeCh4sG27qqv04t_jtAmzee0M/view?usp=sharing
