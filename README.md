# Celebrity-Predictor

This is a Streamlit web application that uses pre-trained deep learning models to recognize faces and recommend similar-looking celebrities. The application uses the InceptionV3 and ResNet50 models to extract features from images, which are then compared to a pre-defined set of features extracted from celebrity images using cosine similarity. The most similar celebrity images are then recommended to the user.

The streamlit library is used to create a web-based GUI, and the mtcnn library is used for face detection. The pre-trained models are loaded using the keras library. The pickle library is used to load pre-computed feature vectors and filenames for the celebrity images. The sklearn library is used for computing the cosine similarity between feature vectors, and the numpy library is used for numerical computations.

The extract_features function takes an image path, a pre-trained model, and a face detector as input and returns a flattened feature vector for the detected face. The function resizes the input image to different scales to handle faces of different sizes and applies the face detector to detect faces. The largest detected face is then used to extract a 224x224 patch, which is then preprocessed and passed through the pre-trained model to extract features.

The recommend function takes a list of pre-computed feature vectors and a feature vector for an input image and returns the indices of the most similar celebrity images. The function computes the cosine similarity, Euclidean distance, and Manhattan distance between the input feature vector and the pre-computed feature vectors and combines them into a weighted score. The indices of the pre-computed feature vectors are then sorted by this score, and the top three indices are returned.

The main_code function takes a feature vector for an input image and a model type as input and returns the predicted celebrity names. The function first calls the recommend function to obtain the indices of the most similar celebrity images, and then uses the filenames associated with these indices to obtain the corresponding celebrity names. The function returns the top three predicted celebrity names.

Finally, the streamlit library is used to create a web-based GUI. The user can upload an image using the file uploader, and the application will display the uploaded image and the predicted celebrity names. The user can choose between two pre-trained models using radio buttons.
