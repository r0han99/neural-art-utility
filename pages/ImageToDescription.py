import pandas as pd
from transformers import pipeline
import streamlit as st 

def load_pipeline():
    try:
        # Create the pipeline for image-to-text
        pipe = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
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
