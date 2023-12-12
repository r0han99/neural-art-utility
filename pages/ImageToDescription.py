import pandas as pd
from transformers import pipeline
import numpy as np
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

st.subheader("Artwork Description Generator", divider='red')

pipe=False
if st.button("Click to Load the Model!"):
    if st.checkbox("Caution! it might break the app, you can always see the model working @huggingface"):
        pipe = load_pipeline()




if pipe:
    uploaded_file = st.file_uploader("Upload an Artwork", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:


        image = Image.open(uploaded_file)
        img_arr = np.array(image)
        img_arr = img_arr/255
        image_place = st.empty()

        image_place.image(image, use_column_width=True)
        try:
            captions = pipe(image)
            image_place.image(image, use_column_width=True, caption=f"{captions[0]['generated_text']}")
        except Exception as e:
            st.error(f"Error in generating caption: {e}")
