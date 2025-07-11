import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
#import SessionState

import numpy as np
model = load_model('Model.h5',compile=False)
model.compile()

st.title('Cat & Dog Image Classifier')
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
        
    
   

