import streamlit as st
import os
import numpy as np
import time
from PIL import Image
import createModel as cm
from spacy import displacy

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://img.freepik.com/free-vector/medical-healthcare-blue-color_1017-26807.jpg?w=1060&t=st=1670813699~exp=1670814299~hmac=0810804041b59cadbd775f26225e42fd9229b448b0103525b29a80feab35cfe3");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

st.title("Chest X-ray Report Generator")
st.markdown("\nThis app will generate report of an X-ray report.\nYou can upload 2 X-rays that are front view and side view of chest of the same patient.")



col1,col2 = st.columns(2)
image_1 = col1.file_uploader("X-ray 1",type=['png','jpg','jpeg'])
image_2 = None
if image_1:
    image_2 = col2.file_uploader("X-ray 2",type=['png','jpg','jpeg'])

col1,col2 = st.columns(2)
predict_button = col1.button('Predict on uploaded files')

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

if predict_button:
    final_features = cm.feature_extraction(image_1,image_2)
    prediction,score = cm.beam_search(final_features,3)
    prediction = prediction.replace('<sos>','')
    predicted_title = '<p style="color:Black; font-size: 20px;font-weight: bold;">Predicted Report:</p>'
    st.markdown(predicted_title, unsafe_allow_html=True)
    st.write(prediction)
    docx = cm.pipeline_NER(prediction)
    html = displacy.render(docx,style="ent")
    NER_predicted_title = '<p style="color:Black; font-size: 20px;font-weight: bold;">NER on Predicted Report:</p>'
    st.markdown(NER_predicted_title, unsafe_allow_html=True)
    st.write(HTML_WRAPPER.format(html),unsafe_allow_html=True)
    #st.write(NER_prediction)
    