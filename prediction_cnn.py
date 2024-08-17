import pickle

from keras.models import Sequential, load_model

import  numpy as np
from tensorflow.keras.utils import img_to_array,load_img

def prediction(image_path):


    model_path = 'cnn_model.h5'
    model = load_model(model_path)

    test_image = load_img(image_path, target_size=(128, 128))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image /= 255
    predictions = model.predict(test_image)

    lb = pickle.load(open('label_transform.pkl', 'rb'))
    print(lb.inverse_transform(predictions)[0])




image_path="test.jpg"

prediction(image_path)