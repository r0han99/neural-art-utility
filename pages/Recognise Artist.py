
import pandas as pd
# from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import streamlit as st 
import matplotlib.pyplot as plt




CODE = """

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

plt.rcParams["font.family"] = "avenir"

# Load the pre-trained model using Keras's load_model function
model = load_model('path_to_your_model.keras')  # Replace with the actual path to your model

# Your class labels
class_labels = {
    0: 'Vincent_van_Gogh',
    1: 'Edgar_Degas',
    2: 'Pablo_Picasso',
    3: 'Pierre-Auguste_Renoir',
    4: 'Albrecht_Dürer',
    5: 'Paul_Gauguin',
    6: 'Francisco_Goya',
    7: 'Rembrandt',
    8: 'Alfred_Sisley',
    9: 'Titian',
    10: 'Marc_Chagall'
}

# Define a function to predict and visualize an image
def predict_and_visualize(image_path, model, class_dictionary):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Make predictions
    predictions = model.predict(img)
    decoded_predictions = decode_predictions(predictions, top=1)[0]

    # Get the predicted class index
    predicted_class_index = decoded_predictions[0][0]

    # Get the predicted label and probability
    predicted_label = class_labels[predicted_class_index]
    prediction_probability = decoded_predictions[0][2]

    # Visualize the image
    plt.imshow(image.load_img(image_path))
    
    # Set the title with the predicted label and probability
    plt.title(f"Prediction: {predicted_label} ({prediction_probability:.2f})")

    # Show the plot
    plt.show()

# Provide the path to the image you want to predict
image_path = 'path_to_your_image.jpg'

# Call the predict_and_visualize function
predict_and_visualize(image_path, model, class_labels)


"""



st.set_page_config(layout='wide', page_icon="🧑🏻‍🎨")
st.subheader("Artwork Description Generator", divider='red')


cols = st.columns([5,3,5])

with cols[0]:
    st.markdown('''<span style="font-size:25px; font-family:menlo">Download InceptionResNet_Final_Tuned.keras</span>''',unsafe_allow_html=True)

with cols[2]:
    st.link_button("Download",url="https://drive.google.com/file/d/17Ji5Yy6xMUyxNhW2nzV5szMuhQLwkfMy/view?usp=drive_link")

st.code(CODE)

st.divider()
st.info("_The trained models are above the usual spatial limit of 100mb to be placed and loaded from the github repository, hence the reason to not include the model inference in the application. Please use this code to run the model at your will._")