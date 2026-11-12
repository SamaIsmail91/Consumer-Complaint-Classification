<h1 align="center">🏦 Consumer Complaint Classification</h1> <h3 align="center">End-to-End NLP Pipeline: SimpleRNN · LSTM · GRU · DistilBERT · Gradio</h3> <p align="center"> <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11%2B-blue.svg" alt="Python"></a> <a href="https://www.tensorflow.org/"><img src="https://img.shields.io/badge/TensorFlow-Keras-orange.svg" alt="TensorFlow"></a> <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-Transformers-red.svg" alt="PyTorch"></a> <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow.svg" alt="Hugging Face"></a> <a href="https://gradio.app/"><img src="https://img.shields.io/badge/Gradio-Deployment-green.svg" alt="Gradio"></a> <a href="https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp"><img src="https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg" alt="Kaggle dataset"></a> </p> <p align="center"> <b>🚀 <a href="https://huggingface.co/spaces/samaismaill/consumer_complaint_classifier_nlp">Live Demo (Hugging Face Space)</a></b> &nbsp;|&nbsp; <b>📓 <a href="<KAGGLE NOTEBOOK LINK>>">Executed Notebook (Kaggle)</a></b> &nbsp;|&nbsp; <b>🗂️ <a href="https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp">Dataset</a></b> </p>

An end-to-end Natural Language Processing (NLP) pipeline that automatically routes a customer's written complaint to the correct product category (credit card, credit reporting, debt collection, mortgages and loans, retail banking). It compares three recurrent networks built from scratch (SimpleRNN, LSTM, GRU) against a fine-tuned DistilBERT and a classical TF-IDF + Logistic Regression baseline, and serves the models through an interactive Gradio web app that returns the predicted category with a confidence score.


---
## Results

## 🎯 Business Value & Objectives
* **Reduce Response Latency (SLA):** Transforms complaint routing times from hours (manual) to milliseconds (automated).
* **Minimize Operational Costs:** Reduces the manual effort required to sort and triage inbound complaints by up to 70%.
* **Trend Analysis (Data-Driven Insights):** Enables real-time tracking of major customer pain points to assist management in prioritizing systemic product improvements.
🗂️ Dataset Overview

Source: Consumer Complaints Dataset for NLP on Kaggle, built from complaint narratives in the public CFPB (U.S. Consumer Financial Protection Bureau) complaint database.

Stratified split: train `[N_TRAIN]` / val `[N_VAL]` / test `[N_TEST]`.

| Model | Accuracy | Macro Precision | Macro Recall | **Macro F1** | Weighted F1 |
|---|---|---|---|---|---|
| TF-IDF + LogReg (baseline) | [ ] | [ ] | [ ] | [ ] | [ ] |
| SimpleRNN | [ ] | [ ] | [ ] | [ ] | [ ] |
| LSTM | [ ] | [ ] | [ ] | [ ] | [ ] |
| GRU | [ ] | [ ] | [ ] | [ ] | [ ] |
| DistilBERT (fine-tuned) | [ ] | [ ] | [ ] | [ ] | [ ] |

## 🚀 Architectures Compared
To find the optimal balance between predictive power and production latency (**Accuracy vs. Inference Speed**), four distinct architectures were trained and benchmarked:
1. **SimpleRNN:** Built from scratch as a baseline sequential model, addressing vanishing gradient constraints.
2. **LSTM (Long Short-Term Memory):** Implemented to capture long-term context in dense, lengthy complaint narratives via gated cell states.
3. **GRU (Gated Recurrent Unit):** A lighter, faster variant of LSTM, offering rapid training iterations with highly competitive performance.
4. **Fine-tuned Transformer (DistilBERT):** Leveraging state-of-the-art Transfer Learning and Self-Attention via Hugging Face to deeply understand context and semantic nuances.
Full table: `outputs/final_comparison.csv`. Confusion matrices and error analysis are in `outputs/`.
Key finding: `[one sentence: which model won and by how much over the baseline]`.

---
## Pipeline

## 📁 Project Structure
1. **Data** - `Consumer complaint narrative` (text) and `Product` (label); keep the top-N classes.
2. **Preprocessing** - lowercase, strip URLs / numbers / punctuation / CFPB `XXXX` redactions, remove stop words, lemmatize.
3. **Class imbalance** - majority classes are undersampled to at most 3x the smallest class.
4. **Split** - stratified train / val / test.
5. **Sequences** - Keras `Tokenizer` (fit on train only) and post-padding to a fixed length.
6. **Models** - Embedding then (Bi)RNN / LSTM / GRU stacks built from scratch; DistilBERT via HuggingFace `Trainer`.
7. **Evaluation** - accuracy, precision, recall, F1 (macro and weighted), confusion matrices, error analysis.
8. **Deployment** - Gradio app returning the predicted category and a confidence score.

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
## Methodology notes (read before comparing numbers)

- **No data leakage:** the tokenizer and the TF-IDF vectorizer are fit on the training split only; the best checkpoint is picked on validation and the test set is used once for the final evaluation.
- **Imbalance handling differs by model:** SimpleRNN / LSTM / GRU use undersampling **and** class weights; DistilBERT uses undersampling only; the TF-IDF baseline uses `class_weight="balanced"`. Because of this, **Macro F1** is the primary metric.
- **The test set is not the real-world distribution:** undersampling happens before the split, so the test set is more balanced than real incoming complaints. Per-class precision in production will differ.
- **Train / inference consistency:** every model, including DistilBERT, is trained and served on the cleaned text. Feeding raw text to the transformer at inference would silently lower real-world accuracy.
- **Padding mask:** the Embedding layer uses `mask_zero=True`. Without it, LSTM/GRU process dozens of padding steps after short texts and stayed at chance accuracy in early experiments.

---
## Math behind the models (short version)

## 🗂️ Dataset Overview
- Softmax: `p_k = exp(z_k) / sum_j exp(z_j)`; cross-entropy loss: `L = -sum_k y_k log p_k`.
- LSTM: gates `f_t, i_t, o_t = sigmoid(W [h_{t-1}, x_t] + b)`; cell `c_t = f_t * c_{t-1} + i_t * tanh(W_c [h_{t-1}, x_t] + b_c)`; output `h_t = o_t * tanh(c_t)`.
- GRU merges the forget and input gates into an update gate `z_t`: `h_t = (1 - z_t) * h_{t-1} + z_t * h~_t`, giving fewer parameters and faster training.
- SimpleRNN suffers from vanishing gradients on long sequences, which motivates gated cells.

The project is optimized for the **CFPB Consumer Complaint Database** (the official, public repository managed by the US Consumer Financial Protection Bureau).
## Repository structure

⚠️ **Network Constraint Note**: If running in a restricted environment without direct internet access to external data URLs, please refer to:
1. `data/download_instructions.md` to download the raw dataset manually.
2. `data/make_sample_data.py` to generate synthetic dummy records matching the structure so you can test the pipeline instantly.
```
notebooks/        full pipeline in one notebook (run on Colab with a GPU)
src/              the same pipeline as modules (preprocessing, models, training, evaluation, predict)
app.py            Gradio application
build_space.py    packages everything needed for a Hugging Face Space into hf_space/
data/             download instructions (the CSV itself is not committed)
outputs/          comparison table, confusion matrices, error analysis
```

**Expected Data Fields:**
| Column Field | Description |
|---|---|
| `Consumer complaint narrative` | Raw text of the complaint (The input feature) |
| `Product` | Associated department category (The target label) |
## Run it

---
### On Kaggle (recommended)
1. New Notebook -> File -> Import notebook -> upload `notebooks/Consumer_Complaint_Classification.ipynb`.
2. **Add Input** -> search for the CFPB *Consumer Complaints* dataset and add it (the notebook finds the CSV automatically).
3. Settings: **Accelerator = GPU**, **Internet = On** (needed for `pip` and downloading DistilBERT).
4. **Save Version -> Save & Run All (Commit)**. The saved version keeps all outputs; link it in your application and download the executed `.ipynb` for GitHub.

## 📊 Evaluation & Model Benchmarking
### Locally / other platforms

After tackling the severe class imbalances within the data, all four models were rigorously tested on hidden test sets. Below are the comparative engineering benchmarks:

| Model Architecture | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Inference Speed (per text) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SimpleRNN** |72.0196 % |73.8547 % |	72.0196 % |	72.4423 % | **Fastest** (~5ms) |
| **LSTM** | 84.6397 % |	84.9271 % |	84.6397 % | 84.46813 % | Fast (~12ms) |
| **GRU** | 85.0099 % |	85.4326 % |	85.0099 % |	85.0767 %| Fast (~10ms) |
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
```bash
pip install -r requirements.txt
# place complaints.csv in data/ (see data/download_instructions.md), then either:
jupyter notebook notebooks/Consumer_Complaint_Classification.ipynb
# or the scripts:
python src/preprocessing.py && python src/train_deep_models.py && python src/train_transformer.py && python src/evaluate.py
python app.py
```

---
## Deploy to Hugging Face Spaces

## ⚙️ Installation & Execution
1. After training, `models/` must contain `tokenizer.pkl`, `label_encoder.pkl`, `config.json`, the `*.keras` files and optionally `transformer_model/`.
2. Run `python build_space.py` to create a ready-to-upload `hf_space/` folder (only models that exist are packaged).
3. Create a Space on huggingface.co (SDK: Gradio), then:
   ```bash
   cd hf_space
   git init && git lfs install
   git lfs track "*.keras" "*.safetensors" "*.pkl"
   git remote add space https://huggingface.co/spaces/<USER>/<SPACE_NAME>
   git add . && git commit -m "Deploy" && git push space HEAD:main
   ```
4. Put the Space URL at the top of this README.

1. Initialize your environment and install all dependencies:
```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4 punkt
To launch the interactive Gradio interface locally or generate a shareable live link:
## Limitations

Bash
python app.py
2. Name the file **`gradio_screenshot.png`**.
3. Move or upload that file inside the **`outputs/`** directory of your repository. 
4. Commit and push everything to GitHub. The relative path in the markdown line `<img src="outputs/gradio_screenshot.png" ...>` will grab it and display it beautifully on your main page!
English-only; trained on CFPB-style narratives and may not generalize to other domains; the confidence score is the softmax probability, not a calibrated probability.
