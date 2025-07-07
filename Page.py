import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

import numpy as np
model = load_model('Model',compile=False)
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
        st.header("We are " + str(probability * 100), "% Sure", ' that Its a Cat')
    else:
        probability = result[0][0]
        st.header("We are " + str(probability * 100), "% Sure", ' that Its a Dog')
    image1 = load_img(input_image)
    image1 = img_to_array(image1)
    image1 = np.array(image1)
    image1 = image1/255.0

    st.image(image1, width=500)

