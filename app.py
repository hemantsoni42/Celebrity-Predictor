from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tensorflow.keras.applications.inception_v3 import InceptionV3
# from tensorflow.keras.applications.inception_v3 import preprocess_input as pI_V3
# from tensorflow.keras.models import Model
import pickle
from sklearn.preprocessing import normalize
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np

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
# model_V3 = InceptionV3(include_top=False, input_shape=(224, 224, 3), pooling='avg')
model_50 = ResNet50(include_top=False, input_shape=(224, 224, 3), pooling='avg')
# feature_list_V3 = pickle.load(open('embedding_V3.pkl', 'rb'))
# filenames_V3 = pickle.load(open('merged_filenames_V3.pkl', 'rb'))
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


# def extract_features(img_path, model_V3, detector):
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     # Define scales for multi-scale detection
#     scales = [0.5, 1.0, 1.5]
#
#     # Initialize empty list for storing features_1
#     features_1 = []
#
#     for scale in scales:
#         # Resize image
#         resized_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
#
#         # Detect faces
#         results = detector.detect_faces(resized_img)
#
#         if not results:  # if no faces are detected
#             continue
#
#         # Get first face detected
#         x, y, width, height = results[0]['box']
#
#         # Convert bounding box coordinates to original scale
#         x = int(x / scale)
#         y = int(y / scale)
#         width = int(width / scale)
#         height = int(height / scale)
#
#         # Expand the bounding box by 10% in all directions
#         x -= int(width * 0.1)
#         y -= int(height * 0.1)
#         width = int(width * 1.2)
#         height = int(height * 1.2)
#
#         # Clip the bounding box to the image dimensions
#         img_height, img_width, _ = img.shape
#         x = max(x, 0)
#         y = max(y, 0)
#         width = min(width, img_width - x)
#         height = min(height, img_height - y)
#
#         # Extract face from original scale image
#         face = img[y:y + height, x:x + width]
#
#         # Resize the image
#         face = cv2.resize(face, (224, 224))
#
#         # Convert to tensor and preprocess
#         face = tf.keras.preprocessing.image.img_to_array(face)
#         face = np.expand_dims(face, axis=0)
#         face = preprocess_input(face)
#
#         # Extract features_1
#         result = model_V3.predict(face).flatten()
#
#         features_1.append(result)
#
#     # Combine features_1 from all scales
#     features_1 = np.concatenate(features_1)
#
#     return features_1

def extract_features(img_path, model, detector, model_value):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    min_size = min(img.shape[:2])
    factor = 0.707  # sqrt(2)/2

    results = []
    for scale in [1.0, factor, factor ** 2]:
        new_size = int(scale * min_size)
        img_resized = cv2.resize(img, (new_size, new_size))

        # detect faces in the resized image
        results_resized = detector.detect_faces(img_resized)

        # rescale the bounding box coordinates to the original image size
        for result in results_resized:
            result['box'] = [int(coord / scale) for coord in result['box']]
            results.append(result)

    if not results:  # if no faces are detected
        return None

    # choose the face with the largest bounding box
    result = max(results, key=lambda x: x['box'][2] * x['box'][3])
    x, y, width, height = result['box']

    # expand the bounding box by 10% in all directions
    x -= int(width * 0.1)
    y -= int(height * 0.1)
    width = int(width * 1.2)
    height = int(height * 1.2)

    # clip the bounding box to the image dimensions
    img_height, img_width, _ = img.shape
    x = max(x, 0)
    y = max(y, 0)
    width = min(width, img_width - x)
    height = min(height, img_height - y)

    face = img[y:y + height, x:x + width]

    # resize the image
    face = cv2.resize(face, (224, 224))

    # convert to tensor and preprocess
    face = tf.keras.preprocessing.image.img_to_array(face)
    face = np.expand_dims(face, axis=0)
    if model_value == 'V3':
        face = pI_V3(face)
    else:
        face = preprocess_input(face)

    # extract features_1
    result = model.predict(face).flatten()

    return result


# def extract_features(img_path, model_V3, detector):
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     results = detector.detect_faces(img)
#
#     if not results:  # if no faces are detected
#         return None
#
#     x, y, width, height = results[0]['box']
#
#     # expand the bounding box by 20% in all directions
#     x -= int(width * 0.2)
#     y -= int(height * 0.2)
#     width = int(width * 1.4)
#     height = int(height * 1.4)
#
#     # clip the bounding box to the image dimensions
#     img_height, img_width, _ = img.shape
#     x = max(x, 0)
#     y = max(y, 0)
#     width = min(width, img_width - x)
#     height = min(height, img_height - y)
#
#     face = img[y:y + height, x:x + width]
#
#     # resize the image
#     face = cv2.resize(face, (224, 224))
#
#     # convert to tensor and preprocess
#     face = tf.keras.preprocessing.image.img_to_array(face)
#     face = np.expand_dims(face, axis=0)
#     face = preprocess_input(face)
#
#     # extract features_1
#     intermediate_layer_model = Model(inputs=model_V3.input, outputs=model_V3.get_layer('global_average_pooling2d').output)
#     result = intermediate_layer_model.predict(face).flatten()
#
#     return result

# def extract_features_50(img_path, model, detector):
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     results = detector.detect_faces(img)
#
#     if not results:  # if no faces are detected
#         return None
#
#     x, y, width, height = results[0]['box']
#
#     # expand the bounding box by 10% in all directions
#     x -= int(width * 0.1)
#     y -= int(height * 0.1)
#     width = int(width * 1.2)
#     height = int(height * 1.2)
#
#     # clip the bounding box to the image dimensions
#     img_height, img_width, _ = img.shape
#     x = max(x, 0)
#     y = max(y, 0)
#     width = min(width, img_width - x)
#     height = min(height, img_height - y)
#
#     face = img[y:y + height, x:x + width]
#
#     # resize the image
#     face = cv2.resize(face, (224, 224))
#
#     # convert to tensor and preprocess
#     face = tf.keras.preprocessing.image.img_to_array(face)
#     face = np.expand_dims(face, axis=0)
#     face = preprocess_input(face)
#
#     # extract features_1
#     result = model.predict(face).flatten()
#
#     return result


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


# def remove_uploaded_image(uploaded_image):
#     try:
#         os.remove(os.path.join('uploads', uploaded_image.name))
#         print("Image removed successfully!")
#         return True
#     except:
#         return False

def main_code(features, model):
    # predicted_actors_1 = [],predicted_actors_2 = []
    if model == 'V3':
        with st.spinner('Finding lookalike celebrities...'):
            indices = recommend(feature_list_V3, features)
        predicted_actors_1 = [" ".join(filenames_V3[index].split('\\')[1].split('_')) for index in indices]
        st.header('Your uploaded image')
        st.image(display_image, width=400)
        st.write("")
        st.header("Top Celebrity Lookalikes")
        images = []
        images_name = []
        for index in indices:
            predicted_actor_1 = " ".join(filenames_V3[index].split('\\')[1].split('_'))
            image_path = 'data\\' + predicted_actor_1 + '.jpg'
            images_name.append(predicted_actor_1)
            images.append(Image.open(image_path))

        col1, col2, col3 = st.columns(3)
        for i in range(len(images)):
            if i % 3 == 0:
                with col1:
                    st.image(images[i], caption=images_name[i], width=200)
            elif i % 3 == 1:
                with col2:
                    st.image(images[i], caption=images_name[i], width=200)
            else:
                with col3:
                    st.image(images[i], caption=images_name[i], width=200)
    elif model == '50':
        with st.spinner('Finding lookalike celebrities...'):
            indices = recommend(feature_list_50, features)

        predicted_actors_2 = [" ".join(filenames_50[index].split('\\')[1].split('_')) for index in indices]
        images = []
        images_name = []
        for index in indices:
            predicted_actor_2 = " ".join(filenames_50[index].split('\\')[1].split('_'))
            image_path = 'data\\' + predicted_actor_2 + '.jpg'
            images_name.append(predicted_actor_2)
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

    # st.header("Is the prediction correct?")
    # selected_actor = st.selectbox("Select the incorrect celebrity", predicted_actors_1+predicted_actors_2)
    # if st.button("Submit Feedback"):
    #     with open('feedback.txt', 'a') as f:
    #         f.write(f'{uploaded_image.name},{selected_actor}\n')
    #         st.success("Thank you for your feedback!")


# incorrect_predictions = []

# define a function to get user feedback and update the recommendations accordingly
# def get_user_feedback(predictions):
#     feedback = st.radio('Is the prediction correct?', ['Yes', 'No'])
#     if feedback == 'No':
#         incorrect_predictions.extend(predictions)
#         new_predictions = recommend(feature_list_V3, features_1, top_n=5)
#         # ignore any celebrities that were previously identified as incorrect
#         new_predictions = [p for p in new_predictions if p not in incorrect_predictions]
#         return new_predictions
#     else:
#         return predictions

st.title(' Celebrity Predictor ')

uploaded_image = st.file_uploader('Choose an image')

# if uploaded_image is not None:
#     if save_uploaded_image(uploaded_image):
#         display_image = Image.open(uploaded_image)
#         with st.spinner('Extracting features_1...'):
#             features_1 = extract_features(os.path.join('uploads',uploaded_image.name),model_V3,detector)
#         if features_1 is None:
#             st.error("No face detected in the uploaded image.")
#         else:
#             with st.spinner('Finding lookalike celebrities...'):
#                 indices = recommend(feature_list_V3, features_1, feedback_indices=st.session_state.feedback_indices)
#             if len(indices) == 0: # If all predictions are incorrect
#                 indices = recommend(feature_list_V3, features_1, top_n=5)
#                 st.warning("All previous predictions were marked incorrect. Here are some new predictions:")
#             predicted_actors = [" ".join(filenames[index].split('\\')[1].split('_')) for index in indices]
#             # Initialize session state variables
#             if 'feedback_indices' not in st.session_state:
#                 st.session_state.feedback_indices = []
#             if 'feedback_is_correct' not in st.session_state:
#                 st.session_state.feedback_is_correct = {}
#             # Create expander for feedback form
#             with st.expander("Provide feedback for predictions"):
#                 for i, index in enumerate(indices):
#                     # Check if prediction has been removed due to incorrect feedback
#                     if index not in st.session_state.feedback_indices:
#                         st.write(f"Prediction {i+1}: {' '.join(filenames[index].split('/')[-1].split('_')[:-1])}")
#                         feedback_form_key = f"feedback_is_correct_{index}"
#                         # Check if feedback form has been submitted
#                         if feedback_form_key not in st.session_state:
#                             st.radio(f"Is prediction {i+1} correct?", ("Yes", "No"), key=feedback_form_key)
#                         else:
#                             st.write(f"Feedback received. Thank you!")
#                             # Remove incorrect predictions
#                             if st.session_state[feedback_form_key] == "No":
#                                 st.session_state.feedback_indices.append(index)
#                     else:
#                         st.write(f"Prediction {i+1}: Removed due to incorrect feedback.")
#             # Filter out incorrect predictions
#             filtered_indices = [index for index in indices if index not in st.session_state.feedback_indices]
#             # If all predictions are incorrect, provide new predictions
#             if len(filtered_indices) == 0:
#                 indices = recommend(feature_list_V3, features_1, top_n=5)
#                 st.warning("All previous predictions were marked incorrect. Here are some new predictions:")
#                 predicted_actors = [" ".join(filenames[index].split('\\')[1].split('_')) for index in indices]
#                 filtered_indices = [index for index in indices if index not in st.session_state.feedback_indices]
#             # Display top lookalikes
#             if len(filtered_indices) > 0:
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.header('Your uploaded image')
#                     st.image(display_image, width=200)
#                 with col2:
#                     st.header("Top Celebrity Lookalikes")
#                     for index in filtered_indices:
#                         predicted_actor = " ".join(filenames[index].split('\\')[1].split('_'))
#                         st.write(f"Prediction: {predicted_actor}")
#                         st.image(load_image(os.path.join('data', predicted_actor, filenames[index])), width=200, caption=predicted_actor)
#
#             else:
#                 st.error("All predictions have been marked incorrect. Please try a different image.")
#                 with st.spinner('Finding new lookalike celebrities...'):
#                     features_1 = extract_features(os.path.join('uploads', uploaded_image.name), model_V3, detector)
#                     indices = recommend(feature_list_V3, features_1, top_n=5)
#                     predicted_actors = [" ".join(filenames[index].split('\\')[1].split('_')) for index in indices]
#                             # Display new predictions
#                     col1, col2 = st.columns(2)
#                     with col1:
#                         st.header('Your uploaded image')
#                         st.image(display_image, width=200)
#                     with col2:
#                         st.header("New Celebrity Lookalikes")
#                         for index in indices:
#                             predicted_actor = " ".join(filenames[index].split('\\')[1].split('_'))
#                             st.write(f"Prediction: {predicted_actor}")
#                             st.image(load_image(os.path.join('data', predicted_actor, filenames[index])), width=200, caption=predicted_actor)
#
#
#             # remove the uploaded image
#             os.remove(os.path.join('uploads', uploaded_image.name))
#             # st.info('Uploaded image removed.')
# else:
#     st.warning('Please upload an image.')

if uploaded_image is not None and save_uploaded_image(uploaded_image):
    # save the image in a directory
    if save_uploaded_image(uploaded_image):
        # load the image
        display_image = Image.open(uploaded_image)
        # extract the features_1
        with st.spinner('Extracting features...'):
#             features_1 = extract_features(os.path.join('uploads', uploaded_image.name), model_V3, detector,'V3')
            features_2 = extract_features(os.path.join('uploads', uploaded_image.name), model_50, detector,'50')

#         if features_1 is None:
#             st.error("No face detected in the uploaded image.")
#         else:
#             main_code(features_1, 'V3')

        if features_2 is None:
            st.error("No face detected in the uploaded image.")
        else:
            # recommend
            main_code(features_2, '50')


else:
    st.warning('Please upload an image.')
