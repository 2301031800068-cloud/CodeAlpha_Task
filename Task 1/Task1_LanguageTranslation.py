import gradio as gr
from deep_translator import GoogleTranslator

# --- 1. Backend Logic ---
def translate_text(text, target_lang):
    if not text or not text.strip():
        return ""
    try:
        translator = GoogleTranslator(source="auto", target=target_lang)
        return translator.translate(text)
    except Exception as e:
        return f"❌ Error: {str(e)}"

# --- 2. New Modern CSS (Glassmorphism & Glow) ---
custom_css = """
#header { 
    text-align: center; 
    padding: 3rem; 
    background: linear-gradient(135deg, #0f172a 0%, #334155 100%);
    border-radius: 20px;
    margin-bottom: 30px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
}
#header h1 {
    color: #38bdf8 !important;
    font-size: 2.5rem !important;
    text-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
}
.input-card, .output-card {
    border: 1px solid #334155 !important;
    border-radius: 15px !important;
    transition: all 0.3s ease;
}
.input-card:focus-within {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.2) !important;
}
"""

# --- 3. UI Construction ---
with gr.Blocks(title="Malay's AI Translator") as demo:
    
    with gr.Column(elem_id="header"):
        gr.Markdown("# 🌐 Malay's Neural Hub")
        gr.Markdown("#### Next-Gen Language Processing | CodeAlpha 2025")

    with gr.Row():
        with gr.Column():
            source_input = gr.Textbox(
                label="Input Terminal",
                placeholder="Enter text to process...",
                lines=8,
                elem_classes=["input-card"]
            )
            
            target_lang = gr.Dropdown(
                label="Target Protocol",
                choices=[
                    ("English", "en"), ("Spanish", "es"), ("French", "fr"), 
                    ("German", "de"), ("Hindi", "hi"), ("Gujarati", "gu"), 
                    ("Japanese", "ja"), ("Korean", "ko"), ("Arabic", "ar")
                ],
                value="en"
            )

        with gr.Column():
            output_display = gr.Textbox(
                label="Neural Output",
                lines=10,
                interactive=False,
                elem_classes=["output-card"]
            )
            
            with gr.Row():
                clear_btn = gr.Button("Reset", variant="secondary")
                manual_btn = gr.Button("Execute Translation", variant="primary")

    # --- Interaction Logic ---
    source_input.change(fn=translate_text, inputs=[source_input, target_lang], outputs=output_display)
    target_lang.change(fn=translate_text, inputs=[source_input, target_lang], outputs=output_display)
    manual_btn.click(fn=translate_text, inputs=[source_input, target_lang], outputs=output_display)
    clear_btn.click(lambda: ["", ""], None, [source_input, output_display])

# --- 4. Run App ---
if __name__ == "__main__":
    # In Gradio 6.0, we pass the CSS and Theme details inside launch()
    demo.launch(
        theme=gr.themes.Monochrome(
            primary_hue="sky", 
            secondary_hue="slate",
            radius_size="lg"
        ),
        css=custom_css
    )