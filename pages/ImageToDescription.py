import pandas as pd
from transformers import pipeline
from PIL import Image
import streamlit as st 

@st.cache_resource
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
    img_arr = Image.open(image)
    img_arr = img_arr / 255.0
    image_place = st.empty()

    if image is not None:
        image_place.image(img_arr, use_column_width=True)
        try:
            captions = pipe(img_arr)
            image_place.image(img_arr, use_column_width=True, caption=f"{captions[0]['generated_text']}")
        except Exception as e:
            st.error(f"Error in generating caption: {e}")
