import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import base64

#import SessionState

import numpy as np
model = load_model('Model.h5',compile=False)
model.compile()

st.title('Cat & Dog Image Classifier')


# Function to encode image to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Apply background image using CSS
def set_background(image_file):
    bin_str = get_base64_of_bin_file(image_file)
    page_bg_img = f'''
    <style>
    .stApp {{
      background-image: url("data:image/jpg;base64,{bin_str}");
      background-size: cover;
      background-position: center;
      background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the background
set_background('p1.jpg')

input_image = st.file_uploader('Upload image')


if st.button('CHECK'):
    predict = load_img(input_image, target_size=(64, 64))
    predict_modified = img_to_array(predict)
    predict_modified = predict_modified / 255
    predict_modified = np.expand_dims(predict_modified, axis=0)
    result = model.predict(predict_modified)
    if result < 0.5:
        probability = 1 - result[0][0]
        if probability<0.60:
            st.write('We are little confused here as we are only ', probability*100, '% sure that its a Cat')
            st.header('Cat 🐱')
            
        else:
            st.write('We are ', probability*100, '% sure that its a Cat')
            st.header('Cat 🐱')
            
    else:
        probability = result[0][0]
        if probability<0.60:
            st.write('We are little confused here as we are only ', probability*100, '% sure that its a Dog')
            st.header('Dog 🐶')
            
        else:
            st.write('We are ', probability*100, '% sure that its a Dog')
            st.header('Dog 🐶')
            
    image1 = load_img(input_image)
    image1 = img_to_array(image1)
    image1 = np.array(image1)
    image1 = image1/255.0

    st.image(image1, width=500)
    st.markdown('Did we guessed it right ?')
    with st.form("key1"):
    # ask for input
        st.write("YES")
        button_check = st.form_submit_button("Good Prediction this time")
    
    with st.form("key2"):
    # ask for input
        st.write("NO")
        button_check = st.form_submit_button("Try again with different picture")
    #if st.button('YES'):
    #    st.write("YAY !!!")
    #if st.button('NO'):
    #    st.write('Sorry for this time, we will try to improve our prediction')
        
    
   

