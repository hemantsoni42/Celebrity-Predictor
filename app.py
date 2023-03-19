from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as pI_V3
import pickle
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np
from itertools import chain

theme = {
    "primaryColor": "#FF4B4B",
    "backgroundColor": "#0E1117",
    "secondaryBackgroundColor": "#262730",
    "textColor": "#FAFAFA",
    "font": "sans-serif"
}

st.set_page_config(page_title="Celebrity Predictor", page_icon=":smiley:", layout="wide",
                   initial_sidebar_state="collapsed")

detector = MTCNN()
model_V3 = InceptionV3(include_top=False, input_shape=(224, 224, 3), pooling='avg')
model_50 = ResNet50(include_top=False, input_shape=(224, 224, 3), pooling='avg')
feature_list_V3 = pickle.load(open('embedding_V3.pkl', 'rb'))
filenames_V3 = pickle.load(open('merged_filenames_V3.pkl', 'rb'))
feature_list_50 = pickle.load(open('embedding.pkl', 'rb'))
filenames_50 = pickle.load(open('merged_filenames_50.pkl', 'rb'))

def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        print("Image saved successfully!")
        return True
    except:
        return False


def extract_features(img_path, model, detector, model_value):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    min_size = min(img.shape[:2])
    factor = 0.707

    results = []
    for scale in [1.0, factor, factor ** 2]:
        new_size = int(scale * min_size)
        img_resized = cv2.resize(img, (new_size, new_size))
        results_resized = detector.detect_faces(img_resized)
        for result in results_resized:
            result['box'] = [int(coord / scale) for coord in result['box']]
            results.append(result)

    if not results:  
        return None

    result = max(results, key=lambda x: x['box'][2] * x['box'][3])
    x, y, width, height = result['box']
    x -= int(width * 0.1)
    y -= int(height * 0.1)
    width = int(width * 1.2)
    height = int(height * 1.2)

    img_height, img_width, _ = img.shape
    x = max(x, 0)
    y = max(y, 0)
    width = min(width, img_width - x)
    height = min(height, img_height - y)

    face = img[y:y + height, x:x + width]

    face = cv2.resize(face, (224, 224))

    face = tf.keras.preprocessing.image.img_to_array(face)
    face = np.expand_dims(face, axis=0)
    if model_value == 'V3':
        face = pI_V3(face)
    else:
        face = preprocess_input(face)

    result = model.predict(face).flatten()

    return result

def recommend(feature_list, features):
    similarity_cos = cosine_similarity(features.reshape(1, -1), feature_list)[0]
    similarity_euclidean = np.linalg.norm(features - feature_list, axis=1)
    similarity_manhattan = np.sum(np.abs(features - feature_list), axis=1)
    similarity_combined = (similarity_cos - np.min(similarity_cos)) / (
            np.max(similarity_cos) - np.min(similarity_cos)) - \
                          (similarity_euclidean / np.max(similarity_euclidean)) - \
                          (similarity_manhattan / np.max(similarity_manhattan))
    indices = np.argsort(similarity_combined)[::-1]

    unique_indices = []
    for i in indices:
        if i not in unique_indices:
            unique_indices.append(i)
        if len(unique_indices) == 3:
            break

    return unique_indices

def main_code(features, model):
    predictions = []
    if model == 'V3':
        with st.spinner('Finding lookalike celebrities...'):
            indices = recommend(feature_list_V3, features)
        predicted_actors_1 = [" ".join(filenames_V3[index].split('\\')[1].split('_')) for index in indices]
        st.header('Your uploaded image')
        st.image(display_image, width=400)
        st.write("")
        st.header("Top Celebrity Lookalikes")
        for index in indices:
            predicted_actor_1 = " ".join(filenames_V3[index].split('\\')[1].split('_'))
            predictions.append(predicted_actor_1)
    elif model == '50':
        with st.spinner('Finding lookalike celebrities...'):
            indices = recommend(feature_list_50, features)

        predicted_actors_2 = [" ".join(filenames_50[index].split('\\')[1].split('_')) for index in indices]


        for index in indices:
            predicted_actor_2 = " ".join(filenames_50[index].split('\\')[1].split('_'))
            predictions.append(predicted_actor_2)
    return predictions

st.title(' Celebrity Predictor ')
uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None and save_uploaded_image(uploaded_image):
    if save_uploaded_image(uploaded_image):
        display_image = Image.open(uploaded_image)
        with st.spinner('Extracting features...'):
            features_1 = extract_features(os.path.join('uploads', uploaded_image.name), model_V3, detector,'V3')
            features_2 = extract_features(os.path.join('uploads', uploaded_image.name), model_50, detector,'50')
        predictions = []
        if features_1 is None:
            st.error("No face detected in the uploaded image.")
        else:
            predictions.append(main_code(features_1, 'V3'))
        if features_2 is None:
            st.error("No face detected in the uploaded image.")
        else:
            predictions.append(main_code(features_2, '50'))
        final_list = list(chain.from_iterable((predictions)))
        print(final_list)
        sorted_predictions = sorted(set(final_list), key=final_list.count, reverse=True)
        images = []
        images_name = []
        for i in sorted_predictions:
          image_path = 'data\\' + i + '.jpg'
          images_name.append(i)
          images.append(Image.open(image_path))

        col4, col5, col6 = st.columns(3)
        for i in range(len(images)):
           if i % 3 == 0:
             with col4:
                st.image(images[i], caption=images_name[i], width=200)
           elif i % 3 == 1:
             with col5:
                st.image(images[i], caption=images_name[i], width=200)
           else:
             with col6:
                st.image(images[i], caption=images_name[i], width=200)
else:
    st.warning('Please upload an image.')
