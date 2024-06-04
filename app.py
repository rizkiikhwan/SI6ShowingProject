import streamlit as st
import numpy as np

import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('models/modelBrainTumorMRI.h5')

labels = open('models/brainTumorLabel.txt', 'r')
labels = labels.read().split('\n')

st.title(f'IRIS testing model-CNN')

uploaded_file = st.file_uploader("Upload File", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    image = tf.io.decode_image(uploaded_file.getvalue(), channels=1, dtype=tf.float32)
    image = tf.image.resize(image, [150, 150])
    image = tf.expand_dims(image, axis=0)
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    prediction = model.predict(image)
    predicted_class = labels[np.argmax(prediction)]

    st.write(f'Predicted: {predicted_class}')
    st.write(f'Confidence: {np.max(prediction * 100, axis=1)[0]:.2f}%')
    st.write(f'All predicted class: ')
    st.write(labels, prediction)
    # st.write(f'{labels}')
    # st.write(f'{prediction}')