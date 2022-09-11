import tensorflow as tf
import numpy as np
from serpapi import GoogleSearch

def predict(directory_p):

    model_path = 'models/brain_tumor.h5'
    img_height=180
    img_width=180
    
    model = tf.keras.models.load_model(model_path)
    img = tf.keras.utils.load_img(directory_p, target_size=(img_height, img_width))

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_label=np.argmax(score)

    brain_tumor = {0:"Glioma Tumor", 1:"Meningioma Tumor", 2:"Not a Tumor", 3:"Pituitary Tumor"}
    prediction = "This image was detected as: {}".format(brain_tumor[class_label])

    params = {
    "engine":"google_scholar",
    "q" : brain_tumor[class_label],
    "api_key" : "b610b0620dbd103d89e0af2f1ee73f6c2f0d8a705164866f6626b0bff36eb9c2"
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results["organic_results"]

    return prediction, organic_results
