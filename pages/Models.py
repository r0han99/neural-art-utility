import pandas as pd
from transformers import pipeline
import streamlit as st 



@st.cache_resource
def load_pipeline():

    # Create the pipeline for image-to-text
    pipe = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

    return pipe


st.set_page_config(layout='wide', page_icon="🧑🏻‍🎨", )

models = st.selectbox("Select a Model", ["Artist Classification", "Artwork Description Generator",], key="models")


if models == "Artwork Description Generator":
    pipe = load_pipeline()

    image = st.file_uploader("Upload an Artwork")
    st.image(image, use_column_width=True)

