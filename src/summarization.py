from transformers import pipeline

def load_summarizer():
    # Cache the model in app.py, but this returns the pipeline
    return pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(pipeline_obj, text):
    """Summarize the text using the provided HF pipeline."""
    if len(text.split()) < 30:
        return "Text is too short to summarize."
        
    try:
        # Let the model naturally decide length with Beam Search for quality
        result = pipeline_obj(text, max_length=130, min_length=30, do_sample=False, num_beams=4, early_stopping=True)
        return result[0]['summary_text'].strip()
    except Exception as e:
        return f"[Summarization Error: {str(e)}]"
