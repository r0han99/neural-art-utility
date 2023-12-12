import streamlit as st
import random 
import os 
import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import scipy



galleria_path = "./assets/galleria/"

def rm_macos_binaries(anylist):

    try:
        anylist.remove(".DS_Store")
    except:
        return anylist
    
    return anylist



def pick_random():

    global galleria_path

    artists = os.listdir(galleria_path)

    # Mac os Caches
    artists = rm_macos_binaries(anylist=artists)

    images_dir = {}
    for artist in artists: 
        images_dir[artist] = random.sample(os.listdir(os.path.join(galleria_path, artist)),3)
                                           
    return images_dir, artists


def fix_name(name):

    return " ".join(name.split("_"))

def plot_artist(artist):

    global galleria_path

    artist_path = os.path.join(galleria_path, artist)
    
    st.markdown(f'''<span style="font-family:avenir; color:orangered; font-size:50px; font-weight:bold;">{fix_name(artist)}</span>''',unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Raw", "Augmented", "Normalised"])

    

    with tab1:
        columns = st.columns(3)
        for i, img in enumerate(images_dir[artist],0):
            img_path = os.path.join(artist_path, img)
            
            # for img dims
            image_arr = Image.open(img_path)
            array = np.array(image_arr)
            
            
            expander = columns[i].expander("Raw Data ~ Pixels", expanded=False)
            columns[i].image(img_path, use_column_width=True, caption="Image Shape {}".format(np.array(image_arr).shape))
            expander.code(array)
    

    with tab2:
        columns = st.columns(3)
        for i, img in enumerate(images_dir[artist],0):
            img_path = os.path.join(artist_path, img)
            
            # for img dims
            image_arr = Image.open(img_path)
            array = np.array(image_arr)

            

            # Reshape the array to fit the generator's requirements
            array = np.expand_dims(array, axis=0)

            # Create an ImageDataGenerator for augmentation
            datagen = ImageDataGenerator(
                validation_split=0.2,
                rescale=1./255.,
                rotation_range=45,

                shear_range=5,
                horizontal_flip=True,
                vertical_flip=True
            )

            # Configure the generator to not apply data augmentation randomly
            datagen.fit(array)

            # Generate augmented data
            augmented_data = next(datagen.flow(array, batch_size=1))

            # Retrieve the augmented array
            augmented_array = augmented_data[0]

            expander = columns[i].expander("Augmented ~ Pixels", expanded=False)
            columns[i].image(augmented_array, caption="Image Shape {}".format(np.array(image_arr).shape))
            expander.code(augmented_array)


    with tab3:
        columns = st.columns(3)
        for i, img in enumerate(images_dir[artist],0):
            img_path = os.path.join(artist_path, img)

            # for img dims
            image_arr = Image.open(img_path)
            array = np.array(image_arr)



            # Normalised 
            expander = columns[i].expander("Normalised ~ Pixels", expanded=False)
            expander.code(array/255.0)
            columns[i].image(array, caption="Image Shape {}".format(np.array(image_arr).shape))
            
            expander.divider()




def show_gallery(images_dir, artists):


    plot_artist(artist=artists[0])
    st.divider()
    plot_artist(artist=artists[1])
    st.divider()
    plot_artist(artist=artists[2])
    st.divider()
        




st.set_page_config(layout='wide', page_icon="🧑🏻‍🎨", )


st.markdown(f'''<span style="font-family:georgia; color:dodgerblue; font-size:80px; font-weight:bold;"><i>Galleria</i></span>''',unsafe_allow_html=True)
st.subheader("Data Preperation",divider="red")


st.warning("Expand to see the Raw, Normalised and Augmented Pixel Data.")
st.subheader("Configuration for Augmentation")
st.code("""
train_datagen = ImageDataGenerator(validation_split=0.2,
                                   rescale=1./255.,
                                   rotation_range=45,
                                   shear_range=5,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                  )

""")

images_dir, artists = pick_random()

st.divider()
show_gallery(images_dir, artists)