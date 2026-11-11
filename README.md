# 🏦 Consumer Complaint Classification

End-to-end NLP pipeline that routes a customer's written complaint to the correct product category.
It compares three recurrent networks built from scratch (SimpleRNN, LSTM, GRU) against a fine-tuned
DistilBERT and a TF-IDF + Logistic Regression baseline, and serves the best model through a Gradio app.

**Live demo:** `<HUGGING FACE SPACE LINK>` | **Executed notebook (with outputs):** `<KAGGLE NOTEBOOK LINK>`

## Dataset

[Consumer Complaints Dataset for NLP](https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp)
(Kaggle), built from the public CFPB Consumer Complaint Database. Rows loaded before balancing: `[N_RAW]`.

- **5 classes:** `credit_card`, `credit_reporting`, `debt_collection`, `mortgages_and_loans`, `retail_banking`
- **After class balancing: 111,670 complaints**, stratified split: train 80,681 / validation 14,238 / test 16,751 (72% / 13% / 15%)
- The narratives in this dataset are already partly cleaned (lowercase, no punctuation), so the cleaning step changes them only slightly.

## Results (held-out test set, 16,751 complaints)

| Model | Accuracy | Macro Precision | Macro Recall | **Macro F1** | Weighted F1 |
|---|---|---|---|---|---|
| TF-IDF + LogReg (baseline) | 0.8580 | 0.8474 | 0.8652 | 0.8555 | 0.8584 |
| SimpleRNN | 0.7571 | 0.7383 | 0.7618 | 0.7479 | 0.7587 |
| LSTM | 0.8456 | 0.8345 | 0.8500 | 0.8411 | 0.8463 |
| GRU | 0.8539 | 0.8422 | 0.8605 | 0.8505 | 0.8542 |
| **DistilBERT (fine-tuned)** | **0.8671** | 0.8629 | 0.8647 | **0.8637** | **0.8670** |

Full table: `outputs/final_comparison.csv`. Confusion matrices and error examples are in the executed notebook.

### What the results say

- **DistilBERT is the best model, but the margin is small:** +0.9 points accuracy and +0.8 points Macro F1 over the TF-IDF baseline. A simple linear model on n-gram features is already strong on this task; the transformer adds a modest gain. These are single runs (one seed) and I did not run a significance test.
- **The from-scratch networks do not beat the baseline.** GRU is closest (-0.5 points Macro F1), LSTM is -1.4 points, and SimpleRNN is about 10 points behind, consistent with plain RNNs struggling to carry information across long sequences (vanishing gradients; not tested separately here). GRU outperformed LSTM by about 1 point in this run, which may not hold across seeds.
- **Most errors come from overlapping categories.** 2,227 of 16,751 test complaints (13.3%) are misclassified by DistilBERT. The two largest confusions are `debt_collection` -> `credit_reporting` (361) and the reverse (348), 32% of all errors together. The ten most frequent confusion pairs include 1,300 errors (58% of all errors) that involve `credit_reporting`. Many misclassified examples are genuinely ambiguous (for example disputes about a collection account that are written as credit-report complaints), so accuracy well above ~87% is unlikely without cleaner labels.

| True class | Predicted as | Count |
|---|---|---|
| debt_collection | credit_reporting | 361 |
| credit_reporting | debt_collection | 348 |
| mortgages_and_loans | credit_reporting | 179 |
| credit_reporting | mortgages_and_loans | 160 |
| credit_card | retail_banking | 140 |

## Pipeline

1. **Data** - complaint narrative (text) and `product` (label).
2. **Preprocessing** - lowercase, strip URLs / numbers / punctuation / `XXXX` redactions, remove stop words, lemmatize.
3. **Class imbalance** - majority classes undersampled to at most 3x the smallest class.
4. **Split** - stratified train / validation / test.
5. **Sequences** - Keras `Tokenizer` (20,000-word vocabulary, fit on train only), post-padding to 150 tokens.
6. **Models** - Embedding (128-d, `mask_zero=True`) -> SpatialDropout(0.2) -> BiRNN/BiLSTM/BiGRU(64) -> RNN/LSTM/GRU(32) -> Dropout(0.4) -> Dense(64, ReLU, L2) -> Dropout(0.3) -> softmax. DistilBERT (`distilbert-base-uncased`) via HuggingFace `Trainer`. Baseline: TF-IDF (50k features, 1-2-grams) + Logistic Regression.
7. **Training setup** - RNNs: Adam (1e-3), batch 64, up to 20 epochs, early stopping on validation loss (patience 5), learning-rate reduction on plateau, class weights. DistilBERT: max length 128, 3 epochs, batch 16, learning rate 2e-5, weight decay 0.01, best checkpoint by validation F1.
8. **Evaluation** - accuracy, precision, recall, F1 (macro and weighted), confusion matrices, error analysis.
9. **Deployment** - Gradio app returning the predicted category and a confidence score.

## Methodology notes (read before comparing numbers)

- **No data leakage:** tokenizer and TF-IDF vectorizer are fit on the training split only; checkpoints are chosen on validation. The test set is used for the final comparison. The overall ranking of the five model families was read off the test set, so treat small gaps as indicative.
- **Imbalance handling differs by model:** SimpleRNN / LSTM / GRU use undersampling **and** class weights; DistilBERT uses undersampling only; the baseline uses `class_weight="balanced"`. That is why **Macro F1** is the primary metric.
- **The test set is not the real-world distribution:** undersampling happens before the split, so the test set is more balanced than incoming complaints. `credit_reporting` is the dominant class in practice, so expect different per-class precision in production.
- **Train / inference consistency:** every model, including DistilBERT, is trained and served on the cleaned text.
- **Padding mask:** the Embedding layer uses `mask_zero=True` so recurrent layers ignore padding. Without it, LSTM/GRU spend dozens of steps on padding after short texts and stayed at chance accuracy in early smoke tests.

## Math behind the models (short version)

- Softmax: `p_k = exp(z_k) / sum_j exp(z_j)`; cross-entropy loss: `L = -sum_k y_k log p_k`.
- LSTM: gates `f_t, i_t, o_t = sigmoid(W [h_{t-1}, x_t] + b)`; cell `c_t = f_t * c_{t-1} + i_t * tanh(W_c [h_{t-1}, x_t] + b_c)`; output `h_t = o_t * tanh(c_t)`.
- GRU merges the forget and input gates into an update gate `z_t`: `h_t = (1 - z_t) * h_{t-1} + z_t * h~_t`, giving fewer parameters and faster training.
- SimpleRNN suffers from vanishing gradients on long sequences, which motivates gated cells.
- TF-IDF weights a term by `tf * log(N / df)` (with sublinear tf); Logistic Regression is a linear softmax classifier on those features.

## Repository structure

```
notebooks/        full pipeline in one notebook (run on Kaggle with a GPU)
src/              the same pipeline as modules (preprocessing, models, training, evaluation, predict)
app.py            Gradio application
build_space.py    packages everything needed for a Hugging Face Space into hf_space/
data/             download instructions (the CSV itself is not committed)
outputs/          comparison table and error analysis
```

## Run it

### On Kaggle (recommended)
1. New Notebook -> File -> Import notebook -> upload `notebooks/Consumer_Complaint_Classification.ipynb`.
2. **Add Input** -> add the dataset linked above (the notebook finds the CSV and the columns automatically).
3. Settings: **Accelerator = GPU**, **Internet = On**.
4. **Save Version -> Save & Run All (Commit)** keeps all outputs.

### Locally
```bash
pip install -r requirements.txt
# place the dataset CSV in data/ (see data/download_instructions.md), then either open the notebook or run:
python src/preprocessing.py && python src/train_deep_models.py && python src/train_transformer.py && python src/evaluate.py
python app.py
```

## Deploy to Hugging Face Spaces

Kaggle's `/kaggle/working` is temporary: files are lost when an interactive session ends unless the notebook was committed (**Save Version -> Save & Run All**). Push the trained artifacts to the Hub from the notebook so they survive.

**Option A - push from Kaggle (recommended)**
1. On huggingface.co create a Space (SDK: Gradio, hardware: CPU basic, Public) and a **Write** access token.
2. In the Kaggle notebook: *Add-ons -> Secrets* -> add `HF_TOKEN`. Then, in the "Back up the trained artifacts" cell, set `PUSH_TO_HUB = True`, `HF_USERNAME` and `SPACE_NAME`, and run **Save & Run All**. The models are uploaded straight into the Space's `models/` folder (training checkpoints are skipped).
3. Download the small file `models/versions.json` from the Kaggle output into your local `models/`, then run `python build_space.py --code-only`.
4. Upload the contents of `hf_space/` (`app.py`, `README.md`, `requirements.txt`, `src/predict.py`) to the same Space (Files -> Add file -> Upload files; keep the `src/` folder).

**Option B - download and upload manually**
1. Download `models/` from the committed version's Output tab (skip `transformer_model/checkpoints/`) into your local `models/`.
2. `python build_space.py`, then upload the contents of `hf_space/` to the Space (browser upload, or `huggingface_hub.upload_folder`).

The first build installs TensorFlow, PyTorch and transformers and takes several minutes; watch the **Logs** tab until the status is *Running*. `build_space.py` pins the library versions recorded in `models/versions.json`, because the pickled tokenizer and `.keras` files are version-sensitive.

## Limitations

English only; the confidence score is the softmax probability, not a calibrated probability; results come from a single training run per model; the test distribution is balanced rather than real-world; class labels are coarse and overlap (especially `credit_reporting` vs `debt_collection`).
