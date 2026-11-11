# 🏦 Consumer Complaint Classification

End-to-end NLP pipeline that routes a customer's written complaint to the correct product category
(credit card, mortgage, debt collection, student loan, ...). It compares three recurrent networks built
from scratch (SimpleRNN, LSTM, GRU) against a fine-tuned DistilBERT and a TF-IDF + Logistic Regression
baseline, and serves the best model through a Gradio web app.

**Live demo:** `<HUGGING FACE SPACE LINK>` | **Notebook:** `notebooks/Consumer_Complaint_Classification.ipynb`
| [Open in Colab](`<COLAB LINK>`)

> ⚠️ Replace every `<...>` and `[ ]` placeholder with numbers from **your own real run**.
> Never publish numbers produced with the synthetic sample data.

## Results

Dataset: CFPB Consumer Complaint Database, `[N_TOTAL]` complaints used, `[N_CLASSES]` classes.
Stratified split: train `[N_TRAIN]` / val `[N_VAL]` / test `[N_TEST]`.

| Model | Accuracy | Macro Precision | Macro Recall | **Macro F1** | Weighted F1 |
|---|---|---|---|---|---|
| TF-IDF + LogReg (baseline) | [ ] | [ ] | [ ] | [ ] | [ ] |
| SimpleRNN | [ ] | [ ] | [ ] | [ ] | [ ] |
| LSTM | [ ] | [ ] | [ ] | [ ] | [ ] |
| GRU | [ ] | [ ] | [ ] | [ ] | [ ] |
| DistilBERT (fine-tuned) | [ ] | [ ] | [ ] | [ ] | [ ] |

Full table: `outputs/final_comparison.csv`. Confusion matrices and error analysis are in `outputs/`.
Key finding: `[one sentence: which model won and by how much over the baseline]`.

## Pipeline

1. **Data** - `Consumer complaint narrative` (text) and `Product` (label); keep the top-N classes.
2. **Preprocessing** - lowercase, strip URLs / numbers / punctuation / CFPB `XXXX` redactions, remove stop words, lemmatize.
3. **Class imbalance** - majority classes are undersampled to at most 3x the smallest class.
4. **Split** - stratified train / val / test.
5. **Sequences** - Keras `Tokenizer` (fit on train only) and post-padding to a fixed length.
6. **Models** - Embedding then (Bi)RNN / LSTM / GRU stacks built from scratch; DistilBERT via HuggingFace `Trainer`.
7. **Evaluation** - accuracy, precision, recall, F1 (macro and weighted), confusion matrices, error analysis.
8. **Deployment** - Gradio app returning the predicted category and a confidence score.

## Methodology notes (read before comparing numbers)

- **No data leakage:** the tokenizer and the TF-IDF vectorizer are fit on the training split only; the best checkpoint is picked on validation and the test set is used once for the final evaluation.
- **Imbalance handling differs by model:** SimpleRNN / LSTM / GRU use undersampling **and** class weights; DistilBERT uses undersampling only; the TF-IDF baseline uses `class_weight="balanced"`. Because of this, **Macro F1** is the primary metric.
- **The test set is not the real-world distribution:** undersampling happens before the split, so the test set is more balanced than real incoming complaints. Per-class precision in production will differ.
- **Train / inference consistency:** every model, including DistilBERT, is trained and served on the cleaned text. Feeding raw text to the transformer at inference would silently lower real-world accuracy.
- **Padding mask:** the Embedding layer uses `mask_zero=True`. Without it, LSTM/GRU process dozens of padding steps after short texts and stayed at chance accuracy in early experiments.

## Math behind the models (short version)

- Softmax: `p_k = exp(z_k) / sum_j exp(z_j)`; cross-entropy loss: `L = -sum_k y_k log p_k`.
- LSTM: gates `f_t, i_t, o_t = sigmoid(W [h_{t-1}, x_t] + b)`; cell `c_t = f_t * c_{t-1} + i_t * tanh(W_c [h_{t-1}, x_t] + b_c)`; output `h_t = o_t * tanh(c_t)`.
- GRU merges the forget and input gates into an update gate `z_t`: `h_t = (1 - z_t) * h_{t-1} + z_t * h~_t`, giving fewer parameters and faster training.
- SimpleRNN suffers from vanishing gradients on long sequences, which motivates gated cells.

## Repository structure

```
notebooks/        full pipeline in one notebook (run on Colab with a GPU)
src/              the same pipeline as modules (preprocessing, models, training, evaluation, predict)
app.py            Gradio application
build_space.py    packages everything needed for a Hugging Face Space into hf_space/
data/             download instructions (the CSV itself is not committed)
outputs/          comparison table, confusion matrices, error analysis
```

## Run it

### On Kaggle (recommended)
1. New Notebook -> File -> Import notebook -> upload `notebooks/Consumer_Complaint_Classification.ipynb`.
2. **Add Input** -> search for the CFPB *Consumer Complaints* dataset and add it (the notebook finds the CSV automatically).
3. Settings: **Accelerator = GPU**, **Internet = On** (needed for `pip` and downloading DistilBERT).
4. **Save Version -> Save & Run All (Commit)**. The saved version keeps all outputs; link it in your application and download the executed `.ipynb` for GitHub.

### Locally / other platforms

```bash
pip install -r requirements.txt
# place complaints.csv in data/ (see data/download_instructions.md), then either:
jupyter notebook notebooks/Consumer_Complaint_Classification.ipynb
# or the scripts:
python src/preprocessing.py && python src/train_deep_models.py && python src/train_transformer.py && python src/evaluate.py
python app.py
```

## Deploy to Hugging Face Spaces

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

## Limitations

English-only; trained on CFPB-style narratives and may not generalize to other domains; the confidence score is the softmax probability, not a calibrated probability.
