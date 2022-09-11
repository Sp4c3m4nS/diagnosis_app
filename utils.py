import tensorflow as tf
import numpy as np
from serpapi import GoogleSearch

def predict(directory_p, scan):

    brain_tumor = {0:"Glioma Tumor", 1:"Meningioma Tumor", 2:"Not a Tumor", 3:"Pituitary Tumor"}
    alzheimers = {0:"Mildly Demented", 1:"Moderately Demented", 2:"Non Demented", 3:"Very Mildly Demented"}
    monkeypox = {0:"Positive for Monkeypox", 1:"Negative for Monkeypox"}
    lung_pneumonia = {0:"Negative for Pneumonia", 1:"Positive for Pneumonia"}
    predict_pointer = {}

    if scan == 'brain_tumor':
        predict_pointer = brain_tumor
    elif scan == 'alzheimers':
        predict_pointer = alzheimers
    elif scan == 'monkeypox':
        predict_pointer = monkeypox
    elif scan == 'lung_pneumonia':
        predict_pointer = lung_pneumonia

    model_path = f'models/{scan}.h5'
    img_height=180
    img_width=180
    
    model = tf.keras.models.load_model(model_path)
    img = tf.keras.utils.load_img(directory_p, target_size=(img_height, img_width))

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_label=np.argmax(score)
    final_label = predict_pointer[class_label]


    prediction = "This image was detected as: {}".format(final_label)

    if not(class_label == 2 or final_label == monkeypox[1] or final_label == lung_pneumonia[0]):
        params = {
        "engine":"google_scholar",
        "q" : final_label,
        "api_key" : "b610b0620dbd103d89e0af2f1ee73f6c2f0d8a705164866f6626b0bff36eb9c2"
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results["organic_results"]
        final_results = []
        for i in range(3):
            final_results.append(organic_results[i]["title"])
            final_results.append(organic_results[i]["link"])

        scholar1 = 'Title: ' + final_results[0] + ' URL: ' + final_results[1]
        scholar2 = 'Title: ' + final_results[2] + ' URL: ' + final_results[3]
        scholar3 = 'Title: ' + final_results[4] + ' URL: ' + final_results[5]
    else:
        scholar1 = ''
        scholar2 = ''
        scholar3 = ''

    return (prediction, scholar1, scholar2, scholar3)
