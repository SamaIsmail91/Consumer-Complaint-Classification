# 🏦 Consumer Complaint Classification — مشروع تصنيف شكاوى العملاء

مشروع NLP كامل (End-to-End) لتصنيف شكاوى العملاء تلقائيًا إلى الفئة الصحيحة (مثل: Credit card, Mortgage, Debt collection, Student loan, ...) باستخدام:

- **SimpleRNN** (مبني من الصفر)
- **LSTM** (مبني من الصفر)
- **GRU** (مبني من الصفر)
- **Fine-tuned Transformer** (DistilBERT عبر HuggingFace)
- **تطبيق Gradio احترافي** للنشر والتجربة الحية مع Confidence Score

---

## 📁 هيكل المشروع

```
consumer_complaint_project/
├── notebooks/
│   └── Consumer_Complaint_Classification.ipynb  all Project Pipline in this notebook)
├── data/
│   ├── download_instructions.md   # طريقة تحميل الداتاست الحقيقي (CFPB)
│   └── make_sample_data.py        # 
├── src/
│   ├── preprocessing.py           # تنظيف النصوص + Tokenization + Padding
│   ├── imbalance.py                # معالجة عدم توازن الفئات (class weights / oversampling)
│   ├── models_rnn.py               # بناء SimpleRNN / LSTM / GRU
│   ├── train_deep_models.py        # تدريب الموديلات الثلاثة وحفظها
│   ├── train_transformer.py        # Fine-tuning لـ DistilBERT
│   ├── evaluate.py                 # مقارنة الأربع موديلات (Accuracy/Precision/Recall/F1/Confusion Matrix)
│   └── predict.py                  # دالة تنبؤ موحدة تستخدمها Gradio
├── models/                          # هنا تُحفظ الموديلات المدربة (.h5 / HF checkpoint)
├── outputs/                         # الرسوم البيانية وتقارير المقارنة
├── app.py                           # تطبيق Gradio الاحترافي
├── requirements.txt
└── README.md
```

---

## 🗂️ الداتا سِت

المشروع مصمم للعمل مع **CFPB Consumer Complaint Database** (الداتاست الرسمي من مكتب حماية المستهلك المالي الأمريكي).

⚠️ **ملاحظة مهمة**: بيئة التنفيذ الحالية هنا لا تملك اتصال إنترنت لتحميل الداتاست الحقيقي مباشرة (الشبكة مقيدة على pypi/github فقط). لذلك:

1. حمّل الداتاست بنفسك من الرابط في `data/download_instructions.md`
2. أو استخدم `data/make_sample_data.py` لتوليد داتا تجريبية (Synthetic) بنفس البنية عشان تجرب الكود كامل وهو شغال قبل ما تجيب الداتا الحقيقية.

الأعمدة المتوقعة في الملف:
| العمود | الوصف |
|---|---|
| `Consumer complaint narrative` | نص الشكوى (الـ Feature) |
| `Product` | فئة الشكوى (الـ Label) |

---

## ⚙️ التثبيت

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4 punkt
```

**يُفضّل التشغيل على Google Colab (GPU مجاني)** خصوصًا لخطوة تدريب الـ Transformer.

---

## 🚀 خطوات التشغيل بالترتيب

### 🟢 الأسهل: نوت بوك واحد يعمل كل حاجة
افتح `notebooks/Consumer_Complaint_Classification.ipynb` في **Google Colab** (يُفضّل تفعيل GPU من
Runtime → Change runtime type)، وشغّل الخلايا بالترتيب من فوق لتحت. كل خطوة موجودة جوه: تثبيت
المكتبات، تحميل/توليد الداتا، التنظيف، بناء وتدريب SimpleRNN/LSTM/GRU، Fine-tuning الـ Transformer،
التقييم والمقارنة، وأخيرًا تشغيل تطبيق Gradio مباشرة جوه النوت بوك (بلينك عام لو `share=True`).

### 🔵 أو: تشغيل عبر ملفات .py منفصلة (للاستخدام في Production / CI)

```bash
# 1) (اختياري) توليد داتا تجريبية للاختبار السريع
python data/make_sample_data.py

# 2) تنظيف ومعالجة النصوص + تقسيم train/val/test
python src/preprocessing.py

# 3) تدريب SimpleRNN / LSTM / GRU
python src/train_deep_models.py

# 4) Fine-tune الـ Transformer (DistilBERT)
python src/train_transformer.py

# 5) تقييم ومقارنة الموديلات الأربعة
python src/evaluate.py

# 6) تشغيل تطبيق Gradio
python app.py
```

---

## 📊 معايير التقييم
Accuracy, Precision (macro/weighted), Recall, F1-score, Confusion Matrix — لكل موديل، بالإضافة إلى جدول مقارنة نهائي في `outputs/model_comparison.csv` ورسم بياني `outputs/comparison_chart.png`.

## 🎨 تطبيق Gradio
واجهة احترافية تعرض:
- خانة إدخال نص الشكوى
- اختيار الموديل (RNN / LSTM / GRU / Transformer)
- الفئة المتوقعة + Confidence Score كنسبة مئوية
- Top-3 احتمالات كـ Bar chart
- تصميم Dark/Light مخصص بـ CSS احترافي
