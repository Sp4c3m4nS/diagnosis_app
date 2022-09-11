import tensorflow as tf
import numpy as np

def predict(directory_p):

    model_path = 'models/brain_tumor.h5'
    img_height=180
    img_width=180
    
    model = tf.keras.models.load_model(model_path)
    img = tf.keras.utils.load_img(
        directory_p, target_size=(img_height, img_width))

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_label=np.argmax(score)

    brain_tumor = {0:"Glioma Tumor", 1:"Meningioma Tumor", 2:"Not a Tumor", 3:"Pituitary Tumor"}

    return print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(brain_tumor[class_label], 100 * np.max(score)))
