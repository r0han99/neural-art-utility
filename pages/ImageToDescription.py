import pandas as pd
from transformers import pipeline
import streamlit as st 


def load_pipeline():

    # Create the pipeline for image-to-text
    pipe = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", from_pt=True)
    return pipe

st.set_page_config(layout='wide', page_icon="🧑🏻‍🎨")

st.title("Artwork Description Generator")
pipe = load_pipeline()

image = st.file_uploader("Upload an Artwork")
image_place = st.empty()

if image is not None:
    image_place.image(image, use_column_width=True)
    captions = pipe(image)
    image_place.image(image, use_column_width=True, caption=f"{captions[0]['generated_text']}")


