# ⚠️ ملاحظة عن الموديلات الموجودة هنا

الموديلات (`simplernn_model.keras`, `lstm_model.keras`, `gru_model.keras`) في هذا المجلد
تم تدريبها على **داتا تجريبية (Synthetic)** تم توليدها بواسطة `data/make_sample_data.py`
فقط لإثبات أن الـ pipeline كامل شغال من أول خطوة لآخر خطوة (بما فيها تطبيق Gradio).

لأن الداتا التجريبية بسيطة جدًا وصغيرة (900 صف / 6 فئات فقط)، الموديلات وصلت لـ 100% accuracy
بسهولة — وهذا **متوقع ولا يعكس أداء حقيقي**.

## للاستخدام الفعلي (Production):
1. حمّل الداتاست الحقيقي من CFPB (راجع `data/download_instructions.md`)
2. شغّل `python src/preprocessing.py` من جديد
3. شغّل `python src/train_deep_models.py` و `python src/train_transformer.py`
4. شغّل `python src/evaluate.py` للمقارنة الحقيقية بين الموديلات
5. الموديلات الجديدة ستحل محل هذه تلقائيًا وتطبيق Gradio هيستخدمها فورًا
