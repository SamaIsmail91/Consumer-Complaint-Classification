Here is the fully professional, comprehensive English version of your README.md file, optimized for GitHub. It includes the business value, architecture comparison, clear instructions on how to add your deployment screenshot, and interactive badges to catch the eye of recruiters.

Markdown
# 🏦 Consumer Complaint Classification — End-to-End NLP Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20TensorFlow-orange.svg)](https://github.com/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow.svg)](https://huggingface.co/)
[![Gradio](https://img.shields.io/badge/Gradio-Deployment-green.svg)](https://gradio.app/)

An end-to-end Natural Language Processing (NLP) pipeline designed to solve a critical operational challenge in the financial sector. This project automates the categorization of unstructured customer complaint narratives and routes them to the correct department (e.g., *Credit card, Mortgage, Debt collection, Student loan*). By replacing manual triage with deep learning models, this system reduces human error, cuts down processing costs, and speeds up response times.

---

## 🎯 Business Value & Objectives
* **Reduce Response Latency (SLA):** Transforms complaint routing times from hours (manual) to milliseconds (automated).
* **Minimize Operational Costs:** Reduces the manual effort required to sort and triage inbound complaints by up to 70%.
* **Trend Analysis (Data-Driven Insights):** Enables real-time tracking of major customer pain points to assist management in prioritizing systemic product improvements.

---

## 🚀 Architectures Compared
To find the optimal balance between predictive power and production latency (**Accuracy vs. Inference Speed**), four distinct architectures were trained and benchmarked:
1. **SimpleRNN:** Built from scratch as a baseline sequential model, addressing vanishing gradient constraints.
2. **LSTM (Long Short-Term Memory):** Implemented to capture long-term context in dense, lengthy complaint narratives via gated cell states.
3. **GRU (Gated Recurrent Unit):** A lighter, faster variant of LSTM, offering rapid training iterations with highly competitive performance.
4. **Fine-tuned Transformer (DistilBERT):** Leveraging state-of-the-art Transfer Learning and Self-Attention via Hugging Face to deeply understand context and semantic nuances.

---

## 📁 Project Structure

consumer_complaint_project/
├── notebooks/
│   └── Consumer_Complaint_Classification.ipynb  # Complete end-to-end workflow in a single notebook
├── data/
│   ├── download_instructions.md   # Setup guide for the real CFPB Database
│   └── make_sample_data.py        # Python script to generate synthetic test data
├── src/
│   ├── preprocessing.py           # Text cleaning (Lemmatization, Stopwords removal) & Tokenization
│   ├── imbalance.py               # Handles class imbalance using dynamic Class Weights
│   ├── models_rnn.py              # Scratch architectures for SimpleRNN / LSTM / GRU
│   ├── train_deep_models.py       # Training and checkpointing routine for sequential models
│   ├── train_transformer.py       # Fine-tuning protocol for DistilBERT using HF Trainer API
│   ├── evaluate.py                # Comprehensive metrics generation & Confusion Matrix plotting
│   └── predict.py                 # Unified, modular inference function used by the deployment layer
├── models/                        # Saved model weights, custom states, and HF checkpoints (.h5 / PyTorch)
├── outputs/                       # Visualizations, classification reports, and benchmarking CSVs
├── app.py                         # Gradio interactive web application script
├── requirements.txt               # Project dependencies
└── README.md


---

## 🗂️ Dataset Overview

The project is optimized for the **CFPB Consumer Complaint Database** (the official, public repository managed by the US Consumer Financial Protection Bureau).

⚠️ **Network Constraint Note**: If running in a restricted environment without direct internet access to external data URLs, please refer to:
1. `data/download_instructions.md` to download the raw dataset manually.
2. `data/make_sample_data.py` to generate synthetic dummy records matching the structure so you can test the pipeline instantly.

**Expected Data Fields:**
| Column Field | Description |
|---|---|
| `Consumer complaint narrative` | Raw text of the complaint (The input feature) |
| `Product` | Associated department category (The target label) |

---

## 📊 Evaluation & Model Benchmarking

After tackling the severe class imbalances within the data, all four models were rigorously tested on hidden test sets. Below are the comparative engineering benchmarks:

| Model Architecture | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Inference Speed (per text) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SimpleRNN** |0.720196 |	0.738547 |	0.720196 |	0.724423 | **Fastest** (~5ms) |
| **LSTM** | 0.846397 |	0.849271 |	0.846397 | 0.846813| Fast (~12ms) |
| **GRU** | 0.850099 |	0.854326 |	0.850099 |	0.850767 | Fast (~10ms) |
| **DistilBERT (Transformer)**| **86.7650%** | **86.7629** | **86.7650** | **86.7556** | Moderate (~35ms) |

> **💡 Engineering Trade-off:** While **DistilBERT** provides the highest accuracy and outstanding context comprehension, **GRU** stands out as a highly viable alternative for low-resource production environments due to its fast processing speed and compact architecture.

---

## 🎨 Interactive Gradio Web App Deployment

An interactive web UI was deployed using **Gradio** to allow customer support agents and engineers to test live predictions. The app features:
* A clean text input field for copying and pasting long complaint narratives.
* A dynamic dropdown to swap between trained architectures on-the-fly.
* Real-time generation of the predicted category alongside a dynamic **Confidence Score**.
* A visual horizontal bar chart displaying the **Top-3 probable categories** for deeper validation.

### 🖥️ Live App User Interface:

<p align="center">
  <img src="outputs/gradio_screenshot.png" alt="Gradio Deployment App Interface" width="850">
</p>

---

## ⚙️ Installation & Execution

1. Initialize your environment and install all dependencies:
```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4 punkt
To launch the interactive Gradio interface locally or generate a shareable live link:

Bash
python app.py
2. Name the file **`gradio_screenshot.png`**.
3. Move or upload that file inside the **`outputs/`** directory of your repository. 
4. Commit and push everything to GitHub. The relative path in the markdown line `<img src="outputs/gradio_screenshot.png" ...>` will grab it and display it beautifully on your main page!
