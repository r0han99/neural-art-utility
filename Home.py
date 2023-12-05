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
        properties = {'font-family':"'avenir'", 'text-align':'center','color':'orangered','font-weight':'regular'}
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
        placeholder.markdown(f'''<span style="font-family:'avenir'; text-align:center; color: white; font-weight:regular; ">{output.strip()}</span>''', unsafe_allow_html=True)  # Display the updated output
        time.sleep(speed)  # Adjust the delay as needed
    
    

# Main function
def main_cs():



    set_google_fonts()

    st.title("Deep Art Inference")
    st.markdown("---")
    

    meta = read_data()

    #st.write(meta)
    st.sidebar.title("Control Shelf")
    st.sidebar.divider()

    artist = st.selectbox("Artist", meta['name'], key='artist-selection')

    input_text = meta[meta['name']==artist]['bio'].values[0]


    

    cols = st.columns([5,1,5])

    cols[0].subheader(artist, divider="rainbow")
    cols[1].markdown("")
    cols[2].image(f'./assets/artists/{artist}.jpg',caption=f'{artist}', width=400)
    with cols[0]:
        simulate_typing(input_text)

if __name__ == "__main__":

    st.set_page_config(layout='wide', page_icon="🧑🏻‍🎨", )
    main_cs()
