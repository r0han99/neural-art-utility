import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForVision2Seq
import streamlit as st 

def load_pipeline():
    try:
        # Explicitly specify the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        model = AutoModelForVision2Seq.from_pretrained("nlpconnect/vit-gpt2-image-captioning", from_pt=True)

        # Create the pipeline
        pipe = pipeline("image-to-text", model=model, tokenizer=tokenizer)
        return pipe
    except Exception as e:
        st.error(f"Failed to load the pipeline: {e}")
        return None

st.set_page_config(layout='wide', page_icon="🧑🏻‍🎨")

st.title("Artwork Description Generator")
pipe = load_pipeline()

if pipe:
    image = st.file_uploader("Upload an Artwork")
    image_place = st.empty()

    if image is not None:
        image_place.image(image, use_column_width=True)
        try:
            captions = pipe(image)
            image_place.image(image, use_column_width=True, caption=f"{captions[0]['generated_text']}")
        except Exception as e:
            st.error(f"Error in generating caption: {e}")
