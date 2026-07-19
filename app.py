"""
app.py
------
تطبيق Gradio احترافي لتصنيف شكاوى العملاء.
يعرض: الفئة المتوقعة + Confidence Score + Top-3 احتمالات كرسم بياني.

التشغيل:
    python app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

from predict import classifier

# --------------------------------------------------------------------------
# Sample complaints for quick demo
# --------------------------------------------------------------------------
EXAMPLES = [
    ["I was charged twice for the same purchase on my credit card and the bank refuses to refund me despite multiple calls.",
     "SimpleRNN"],
    ["My mortgage servicer started foreclosure even though I have been making my payments on time every month.",
     "LSTM"],
    ["A debt collector keeps calling me multiple times a day about a debt that isn't even mine and won't stop.",
     "GRU"],
    ["My student loan servicer misapplied my payments and now my balance is higher than it should be.",
     "Transformer (DistilBERT)"],
]

CUSTOM_CSS = """
:root {
    --brand-blue: #2563eb;
    --brand-dark: #0f172a;
}
.gradio-container {
    font-family: 'Segoe UI', 'Inter', sans-serif !important;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
}
#header {
    text-align: center;
    padding: 28px 16px 10px 16px;
}
#header h1 {
    font-size: 2.1rem;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    margin-bottom: 4px;
}
#header p {
    color: #94a3b8;
    font-size: 1.02rem;
}
.panel-card {
    background: rgba(30, 41, 59, 0.75) !important;
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    border-radius: 16px !important;
    padding: 6px !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}
#predict-btn {
    background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
    color: white !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1.05rem !important;
}
#predict-btn:hover {
    filter: brightness(1.1);
    transform: translateY(-1px);
}
#result-label {
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    text-align: center;
}
#confidence-box {
    text-align: center;
    font-size: 1.1rem;
    color: #34d399;
    font-weight: 700;
}
footer {visibility: hidden}
"""

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="violet",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_width="1px",
    button_primary_background_fill="linear-gradient(90deg, #2563eb, #7c3aed)",
)


def plot_top_k(top_results):
    labels = [r[0] for r in top_results][::-1]
    scores = [r[1] * 100 for r in top_results][::-1]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    colors = plt.cm.Blues([0.9, 0.65, 0.4])[::-1] if len(scores) == 3 else plt.cm.Blues(
        [0.9 - 0.25 * i for i in range(len(scores))]
    )
    bars = ax.barh(labels, scores, color=colors)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                 f"{score:.1f}%", va="center", color="#e2e8f0", fontsize=10, fontweight="bold")

    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence (%)", color="#94a3b8")
    ax.tick_params(colors="#e2e8f0")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#475569")
    ax.set_title("Top Predictions", color="#e2e8f0", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def classify_complaint(text, model_name):
    if not text or not text.strip():
        return "⚠️ يرجى إدخال نص الشكوى", "", None

    label, confidence, top_results = classifier.predict(text, model_name, top_k=3)

    if label is None:
        return "⚠️ لم يتم التعرف على النص", "", None

    result_html = f"""
    <div style='padding:14px; border-radius:12px; background:rgba(37,99,235,0.12); border:1px solid rgba(37,99,235,0.35);'>
        <div style='color:#93c5fd; font-size:0.85rem; margin-bottom:4px;'>الفئة المتوقعة / Predicted Category</div>
        <div style='color:#f8fafc; font-size:1.5rem; font-weight:800;'>{label}</div>
    </div>
    """
    confidence_html = f"""
    <div style='margin-top:10px; text-align:center; color:#34d399; font-size:1.15rem; font-weight:700;'>
        ✅ Confidence Score: {confidence*100:.2f}%
    </div>
    """
    fig = plot_top_k(top_results)
    return result_html, confidence_html, fig


# --------------------------------------------------------------------------
# Build UI
# --------------------------------------------------------------------------
with gr.Blocks(theme=THEME, css=CUSTOM_CSS, title="Consumer Complaint Classifier") as demo:
    with gr.Column(elem_id="header"):
        gr.Markdown("# 🏦 Consumer Complaint Classifier")
        gr.Markdown("نظام ذكاء اصطناعي لتصنيف شكاوى العملاء تلقائيًا | Automatic complaint routing powered by Deep Learning & Transformers")

    with gr.Row():
        with gr.Column(scale=1, elem_classes="panel-card"):
            gr.Markdown("### ✍️ أدخل نص الشكوى")
            complaint_input = gr.Textbox(
                label="Complaint narrative",
                placeholder="Type or paste the customer complaint here...",
                lines=8,
            )
            model_choice = gr.Radio(
                choices=classifier.available_models,
                value=classifier.available_models[0],
                label="🧠 اختر الموديل",
            )
            predict_btn = gr.Button("🔍  Classify Complaint", elem_id="predict-btn", size="lg")

            gr.Examples(
                examples=EXAMPLES,
                inputs=[complaint_input, model_choice],
                label="💡 أمثلة سريعة",
            )

        with gr.Column(scale=1, elem_classes="panel-card"):
            gr.Markdown("### 📊 النتيجة")
            result_output = gr.HTML()
            confidence_output = gr.HTML()
            chart_output = gr.Plot(label="Top-3 Predictions")

    predict_btn.click(
        fn=classify_complaint,
        inputs=[complaint_input, model_choice],
        outputs=[result_output, confidence_output, chart_output],
    )
    complaint_input.submit(
        fn=classify_complaint,
        inputs=[complaint_input, model_choice],
        outputs=[result_output, confidence_output, chart_output],
    )

    gr.Markdown(
        "<div style='text-align:center; color:#64748b; margin-top:24px; font-size:0.85rem;'>"
        "Built with TensorFlow, HuggingFace Transformers & Gradio — Consumer Complaint Classification Project"
        "</div>"
    )


if __name__ == "__main__":
    demo.launch()
