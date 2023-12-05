import streamlit as st
import time
import sys 
import os
import pandas as pd 



# Function to read data
@st.cache_resource
def read_data():
    meta = pd.read_csv("./data/artists.csv")
    return meta

# Function to set up Google Fonts
def set_google_fonts():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
            body {
                font-family: 'Poppins', sans-serif;
            }
        </style>

        """,
        unsafe_allow_html=True,
    )

def format_text(text, properties:dict = None):
    
    if properties == None:
        properties = {'font-family':"'avenir'", 'text-align':'center','color':'orangered','font-weight':'regular',"font-size":"50px"}
        style = ""
        for attr, value in zip(properties.keys(), properties.values()):
            style += attr + ":" + value + '; '
        

    # st.code(style)
    
    st.markdown(f'''<span style="{style}">{text}</span>''', unsafe_allow_html=True)






def simulate_typing(text, speed=0.00256):


    placeholder = st.empty()
    output = ""
    for char in text:
        output += char + ""
        placeholder.markdown(f'''<span style="font-family:'avenir'; text-align:center; color: white; font-weight:regular; font-size:22px; ">{output.strip()}</span>''', unsafe_allow_html=True)  # Display the updated output
        time.sleep(speed)  # Adjust the delay as needed
    
    

# Main function
def main_cs():


    st.markdown(f'''<center><span style="font-family:georgia; color:orangered; font-size:90px; font-weight:bold;"><i>Deep Art Inference</i></span></center>''',unsafe_allow_html=True)
    st.divider()
    

    meta = read_data()

    #st.write(meta)
    st.sidebar.markdown(f'''<span style="font-family:Avenir; color:white; font-size:35px; font-weight:bold; ">The Idea</span>''',unsafe_allow_html=True)
    about = '''Enter our creative playground! My software blends cutting-edge technology with creative discovery. My canvas encourages you to explore art via a variety of unique methods. Ever wondered who created a masterpiece? Model-1 uses machine learning classification to properly identify artists by examining styles, color palettes, and brush strokes. Enter Model-2, a captivating Siamese Network that compares art to discover how it blends. My Art-to-Description Model-3 Transformer uses words to create colorful descriptions of artworks. Check out Model-4, where Neural Style Transfer and GANs create new creative forms by transferring styles or creating originals. Technology and art combine in the most interesting and surprising ways on this amazing voyage!'''
    st.sidebar.markdown(f'''<span style="font-family:georgia; color:orange; text-align: center; font-size:16px; font-style: italic; ">{about}</span>''',unsafe_allow_html=True)
    st.sidebar.divider()


    st.markdown(f'''<span style="font-family:georgia; color:orange; font-size:50px; font-weight:bold; "><i>Know the Artist </i>🧑🏻‍🎨</span>''',unsafe_allow_html=True)
    st.markdown("")

    artist = st.selectbox("Artist", meta['name'], key='artist-selection')
    input_text = meta[meta['name']==artist]['bio'].values[0]
    cols = st.columns([5,1,7])
    cols[0].subheader(artist, divider="rainbow")
    cols[1].markdown("")
    cols[2].image(f'./assets/artists/{artist}.jpg',caption=f'{artist}', width=500)
    with cols[0]:
        simulate_typing(input_text)

if __name__ == "__main__":

    st.set_page_config(layout='wide', page_icon="🧑🏻‍🎨", )
    main_cs()
