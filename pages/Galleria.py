import streamlit as st
import random 
import os 
import numpy as np
from PIL import Image



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

    columns = st.columns(3)
    for i, img in enumerate(images_dir[artist],0):
        img_path = os.path.join(artist_path, img)
        
        # for img dims
        image_arr = Image.open(img_path)
        

        columns[i].image(img_path, use_column_width=True, caption="Image Shape {}".format(np.array(image_arr).shape))
        expander = columns[i].expander("Raw Data ~ Pixels", expanded=False)
        expander.code(np.array(image_arr)[:10])



def show_gallery(images_dir, artists):


    plot_artist(artist=artists[0])
    st.divider()
    plot_artist(artist=artists[1])
    st.divider()
    plot_artist(artist=artists[2])
    st.divider()
        




st.set_page_config(layout='wide', page_icon="🧑🏻‍🎨", )


st.markdown(f'''<span style="font-family:georgia; color:dodgerblue; font-size:80px; font-weight:bold;"><i>Galleria</i></span>''',unsafe_allow_html=True)
st.divider()

images_dir, artists = pick_random()
show_gallery(images_dir, artists)