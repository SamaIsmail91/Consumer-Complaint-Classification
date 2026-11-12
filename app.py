"""
app.py
------
تطبيق Gradio احترافي لتصنيف شكاوى العملاء.
يعرض: الفئة المتوقعة + Confidence Score + Top-3 احتمالات كرسم بياني.

التشغيل:
    python app.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["OMP_NUM_THREADS"] = "1"

import time
import spaces
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import inspect
import gradio as gr
import matplotlib.pyplot as plt
from predict import classifier

EXAMPLES = [
    ["I was charged twice for the same purchase on my credit card and the bank refuses to refund me despite multiple calls."],
    ["My mortgage servicer started foreclosure even though I have been making my payments on time every month."],
    ["A debt collector keeps calling me multiple times a day about a debt that isn't even mine and won't stop."],
    ["My student loan servicer misapplied my payments and now my balance is higher than it should be."],
]

CUSTOM_CSS = """
@import url('https://googleapis.com');
body, .gradio-container {
    font-family: 'Inter', system-ui, sans-serif !important;
    background-color: #030712 !important;
    background-image: 
        radial-gradient(at 0% 0%, rgba(79, 70, 229, 0.18) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(14, 165, 233, 0.15) 0px, transparent 50%),
        linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px) !important;
    background-size: 100% 100%, 100% 100%, 32px 32px, 32px 32px !important;
    color: #f8fafc !important;
}
.hero-wrap {
    position: relative;
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.85) 0%, rgba(9, 13, 22, 0.95) 100%);
    border: 1px solid rgba(99, 102, 241, 0.28);
    border-radius: 24px;
    padding: 32px 24px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.08);
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 14px;
    border-radius: 999px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #34d399;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    margin-bottom: 12px;
}
.dot-live {
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 8px #10b981;
}
.hero-h1 {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    background: linear-gradient(135deg, #ffffff 20%, #a5b4fc 60%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 10px 0 !important;
}
.hero-desc {
    color: #94a3b8 !important;
    font-size: 1.02rem !important;
    max-width: 660px;
    margin: 0 auto 18px auto !important;
    line-height: 1.6;
}
.metrics-bar {
    display: flex;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
}
.metric-chip {
    background: rgba(30, 41, 59, 0.55);
    border: 1px solid rgba(148, 163, 184, 0.16);
    padding: 7px 15px;
    border-radius: 12px;
    font-size: 0.82rem;
    color: #e2e8f0;
    font-weight: 600;
}
.pro-panel {
    background: rgba(15, 23, 42, 0.72) !important;
    border: 1px solid rgba(148, 163, 184, 0.14) !important;
    border-radius: 20px !important;
    padding: 20px !important;
    backdrop-filter: blur(16px);
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4) !important;
}
textarea {
    background: rgba(3, 7, 18, 0.75) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-radius: 14px !important;
    color: #f8fafc !important;
    font-size: 0.98rem !important;
    line-height: 1.6 !important;
}
textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.22) !important;
}
#predict-btn, #compare-btn {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 50%, #0ea5e9 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 22px !important;
    box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.55) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
#predict-btn:hover, #compare-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 14px 30px -5px rgba(14, 165, 233, 0.6) !important;
}
footer {visibility: hidden !important;}
"""

THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="sky", neutral_hue="slate").set(
    body_background_fill="#030712",
    block_background_fill="rgba(15, 23, 42, 0.72)",
)
def get_category_icon(label):
    lbl = label.lower()
    if "credit" in lbl or "card" in lbl:
        return "💳"
    if "mortgage" in lbl or "loan" in lbl:
        return "🏠"
    if "debt" in lbl or "collection" in lbl:
        return "📞"
    if "bank" in lbl or "account" in lbl:
        return "🏦"
    return "📂"


def plot_top_k(top_results):
    labels = [r[0] for r in top_results][::-1]
    scores = [r[1] * 100 for r in top_results][::-1]

    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    palette = ["#334155", "#6366f1", "#38bdf8"][-len(scores):]
    bars = ax.barh(labels, scores, color=palette, height=0.52)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 2,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}%",
            va="center",
            color="#f8fafc",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlim(0, 114)
    ax.set_xlabel("Confidence Probability (%)", color="#94a3b8", fontsize=9)
    ax.tick_params(axis="y", colors="#f1f5f9", labelsize=10)
    ax.tick_params(axis="x", colors="#64748b", labelsize=8.5)
    ax.grid(axis="x", linestyle=":", alpha=0.2, color="#94a3b8")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#1e293b")
    ax.set_title("Probability Distribution (Top-3)", color="#cbd5e1", fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    return fig


def get_placeholder_html():
    return (
        '<div style="text-align:center;padding:52px 24px;border:1px dashed rgba(99,102,241,0.3);border-radius:18px;background:rgba(3,7,18,0.45);">'
        '<div style="font-size:2.6rem;margin-bottom:10px;">⚡</div>'
        '<div style="color:#f8fafc;font-weight:700;font-size:1.12rem;margin-bottom:6px;">Awaiting Complaint Narrative</div>'
        '<div style="color:#94a3b8;font-size:0.9rem;max-width:340px;margin:0 auto;">Paste a customer complaint on the left or select a test scenario to generate instant AI routing insights.</div>'
        '</div>'
    )


@spaces.GPU
def classify_complaint(text, model_name):
    if not text or not text.strip():
        return '<div style="padding:14px;border-radius:12px;background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.35);color:#fde68a;">⚠️ Please enter a customer complaint first.</div>', None
    if not model_name:
        return '<div style="padding:14px;border-radius:12px;background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.35);color:#fecaca;">⚠️ No models found in models/ folder.</div>', None

    t0 = time.time()
    try:
        label, confidence, top_results = classifier.predict(text, model_name, top_k=3)
    except Exception as e:
        return f'<div style="padding:14px;border-radius:12px;background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.35);color:#fecaca;">❌ Error: {type(e).__name__}: {e}</div>', None
    ms = (time.time() - t0) * 1000

    if label is None:
        return '<div style="padding:14px;border-radius:12px;background:rgba(245,158,11,0.12);color:#fde68a;">⚠️ No valid words found after text preprocessing.</div>', None

    pct = confidence * 100
    word_count = len(text.split())
    icon = get_category_icon(label)

    if pct >= 80:
        tier_text, tier_col, tier_bg = "HIGH CONFIDENCE • AUTO-ROUTE", "#10b981", "rgba(16,185,129,0.15)"
    elif pct >= 55:
        tier_text, tier_col, tier_bg = "MODERATE CONFIDENCE", "#f59e0b", "rgba(245,158,11,0.15)"
    else:
        tier_text, tier_col, tier_bg = "LOW CONFIDENCE • HUMAN REVIEW", "#ef4444", "rgba(239,68,68,0.15)"

    card = (
        '<div style="background:linear-gradient(160deg,rgba(30,41,59,0.85),rgba(9,13,22,0.95));border:1px solid rgba(99,102,241,0.4);border-radius:18px;padding:22px;box-shadow:0 15px 35px rgba(0,0,0,0.4);">'
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px;">'
        f'<span style="font-size:0.72rem;background:{tier_bg};color:{tier_col};border:1px solid {tier_col}44;padding:4px 10px;border-radius:999px;font-weight:800;letter-spacing:0.05em;">{tier_text}</span>'
        f'<span style="font-size:0.76rem;color:#94a3b8;font-family:monospace;">🧠 {model_name} • ⚡ {ms:.0f}ms • 📝 {word_count} words</span>'
        '</div>'
        f'<div style="font-size:1.65rem;font-weight:800;color:#ffffff;margin-bottom:18px;line-height:1.3;">{icon} {label}</div>'
        '<div style="display:flex;justify-content:space-between;font-size:0.86rem;margin-bottom:6px;">'
        '<span style="color:#94a3b8;font-weight:600;">Confidence Score</span>'
        f'<span style="color:{tier_col};font-weight:800;font-size:0.95rem;">{pct:.2f}%</span>'
        '</div>'
        '<div style="width:100%;height:10px;background:#030712;border-radius:999px;overflow:hidden;border:1px solid rgba(255,255,255,0.08);">'
        f'<div style="width:{min(pct,100):.1f}%;height:100%;background:linear-gradient(90deg,#4f46e5,#38bdf8,{tier_col});border-radius:999px;"></div>'
        '</div></div>'
    )
    return card, plot_top_k(top_results)


@spaces.GPU
def compare_all_models(text):
    if not text or not text.strip():
        return '<div style="color:#fde68a;padding:12px;">⚠️ Please enter complaint text first.</div>'
    results = []
    for m in classifier.available_models:
        t0 = time.time()
        try:
            lbl, conf, _ = classifier.predict(text, m, top_k=1)
            dt = (time.time() - t0) * 1000
            results.append((m, lbl, conf * 100, dt, None))
        except Exception as e:
            results.append((m, None, 0, 0, type(e).__name__))

    valid_confs = [r[2] for r in results if r[4] is None]
    best_conf = max(valid_confs) if valid_confs else 0

    rows = ""
    for m, lbl, pct, dt, err in results:
        if err:
            rows += f'<tr><td style="padding:14px;color:#fff;font-weight:700;">🧠 {m}</td><td colspan="3" style="padding:14px;color:#fca5a5;">Error: {err}</td></tr>'
        else:
            is_best = (pct == best_conf and pct > 0)
            badge = '<span style="margin-left:8px;font-size:0.7rem;background:rgba(16,185,129,0.2);color:#34d399;padding:2px 8px;border-radius:999px;border:1px solid rgba(16,185,129,0.4);">🏆 TOP MATCH</span>' if is_best else ""
            row_bg = "rgba(99,102,241,0.08)" if is_best else "transparent"
            rows += (
                f'<tr style="border-bottom:1px solid rgba(148,163,184,0.1);background:{row_bg};">'
                f'<td style="padding:14px;font-weight:700;color:#fff;">🧠 {m}{badge}</td>'
                f'<td style="padding:14px;color:#e2e8f0;font-weight:600;">{get_category_icon(lbl)} {lbl}</td>'
                f'<td style="padding:14px;color:#38bdf8;font-weight:800;">{pct:.2f}%</td>'
                f'<td style="padding:14px;color:#94a3b8;font-family:monospace;">{dt:.0f} ms</td>'
                '</tr>'
            )
    return (
        '<div style="overflow-x:auto;border-radius:16px;border:1px solid rgba(99,102,241,0.25);background:rgba(9,13,22,0.85);">'
        '<table style="width:100%;border-collapse:collapse;text-align:left;">'
        '<thead><tr style="background:rgba(30,41,59,0.85);color:#94a3b8;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em;">'
        '<th style="padding:14px;">Neural Architecture</th><th style="padding:14px;">Predicted Category</th><th style="padding:14px;">Confidence</th><th style="padding:14px;">Latency</th>'
        f'</tr></thead><tbody>{rows}</tbody></table></div>'
    )


_NEW = "theme" in inspect.signature(gr.Blocks.launch).parameters
_b_kw = {} if _NEW else {"theme": THEME, "css": CUSTOM_CSS}
_l_kw = {"theme": THEME, "css": CUSTOM_CSS} if _NEW else {}

with gr.Blocks(title="Consumer Complaint AI Classifier", **_b_kw) as demo:
    gr.HTML(
        '<div class="hero-wrap">'
        '<div style="display:inline-flex;align-items:center;gap:8px;padding:6px 16px;border-radius:999px;background:rgba(16,185,129,0.2);border:1px solid #10b981;color:#6ee7b7 !important;font-size:0.8rem;font-weight:800;letter-spacing:0.06em;margin-bottom:14px;box-shadow:0 0 15px rgba(16,185,129,0.25);"><span style="width:9px;height:9px;background:#34d399;border-radius:50%;box-shadow:0 0 10px #34d399;display:inline-block;"></span><span style="color:#6ee7b7 !important;">SYSTEM ONLINE • GPU ACCELERATED</span></div>'
        '<h1 class="hero-h1">Consumer Complaint AI Classifier</h1>'
        '<p class="hero-desc">Enterprise-grade NLP ticket routing engine. Automatically classifies financial customer disputes with real-time probability analytics and multi-model benchmarking.</p>'
        '<div class="metrics-bar">'
        f'<div class="metric-chip">🧠 Models Loaded: <span style="color:#818cf8;">{len(classifier.available_models)}</span></div>'
        '<div class="metric-chip">⚙️ Pipeline: <span style="color:#38bdf8;">NLTK + Deep RNNs</span></div>'
        '<div class="metric-chip">📊 Output: <span style="color:#34d399;">Top-3 Probabilities</span></div>'
        '</div></div>'
    )

    with gr.Tabs():
        with gr.Tab("🎯 Real-Time Classifier"):
            with gr.Row():
                with gr.Column(scale=5, elem_classes="pro-panel"):
                    gr.Markdown("### ✍️ Customer Complaint Narrative")
                    inp = gr.Textbox(label="Complaint Text", placeholder="Type or paste a financial customer complaint here...", lines=6)
                    mod = gr.Radio(choices=classifier.available_models, value=classifier.available_models if classifier.available_models else None, label="🧠 Select Neural Architecture")
                    with gr.Row():
                        gr.ClearButton(components=[inp], value="🗑️ Clear", variant="secondary")
                        btn = gr.Button("🚀 Analyze & Classify", elem_id="predict-btn", scale=2)
                    gr.Examples(examples=EXAMPLES, inputs=[inp], label="💡 Sample Test Cases")

                with gr.Column(scale=5, elem_classes="pro-panel"):
                    gr.Markdown("### 📈 Diagnostic Insights")
                    out_html = gr.HTML(value=get_placeholder_html())
                    out_plot = gr.Plot(label="Confidence Breakdown")

        with gr.Tab("⚖️ Architecture Benchmark"):
            with gr.Column(elem_classes="pro-panel"):
                gr.Markdown("### 🔬 Multi-Model Side-by-Side Comparison")
                gr.Markdown("Evaluate all available neural networks simultaneously on the same complaint to compare accuracy confidence and inference speed.")
                c_inp = gr.Textbox(label="Benchmark Complaint Narrative", placeholder="Enter a complaint to test across all models...", lines=4)
                c_btn = gr.Button("⚡ Run Multi-Model Benchmark", elem_id="compare-btn")
                c_out = gr.HTML()

    btn.click(fn=classify_complaint, inputs=[inp, mod], outputs=[out_html, out_plot])
    inp.submit(fn=classify_complaint, inputs=[inp, mod], outputs=[out_html, out_plot])
    c_btn.click(fn=compare_all_models, inputs=[c_inp], outputs=[c_out])

    gr.Markdown(
        "<div style='text-align:center; color:#64748b; margin-top:24px; font-size:0.85rem;'>"
        "Built with TensorFlow, HuggingFace Transformers & Gradio — Consumer Complaint Classification Project"
        "</div>"
    )

if __name__ == "__main__":
    demo.launch(**_l_kw)
