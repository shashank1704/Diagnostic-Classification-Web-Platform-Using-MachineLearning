

import os
import sys
import numpy as np
import operator
import pickle
from keras.models import Sequential, load_model
import cv2
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image as image_utils
import numpy
from keras.preprocessing import image
from tensorflow.keras.utils import load_img
from tensorflow.keras.applications.vgg16 import VGG16
def classify_image(test_image):
    try:

        test_images = []

        img_path = test_image

        testing_img = cv2.imread(img_path)

        cv2.imwrite("../MLProject/static/detection.jpg", testing_img)

        test_image = load_img(test_image, target_size=(128, 128))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image /= 255

        test_image = test_image.reshape(len(test_image), -1)


        RF_model = pickle.load(open('rf.model', 'rb'))

        prediction_res = RF_model.predict(test_image)

        print(prediction_res)



        return prediction_res[0]





    except Exception as e:
        print("Error=", e)
        tb = sys.exc_info()[2]
        print("LINE NO: ", tb.tb_lineno)




def classify_image_cnn(test_image):
    try:

        test_images = []

        img_path = test_image

        testing_img = cv2.imread(img_path)

        cv2.imwrite("../MLProject/static/detection.jpg", testing_img)

        model_path = 'cnn_model.h5'
        model = load_model(model_path)

        test_image = load_img(test_image, target_size=(128, 128))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image /= 255
        predictions = model.predict(test_image)

        lb = pickle.load(open('label_transform.pkl', 'rb'))
        print(lb.inverse_transform(predictions)[0])

        pred_res = lb.inverse_transform(predictions)[0]


        return pred_res





    except Exception as e:
        print("Error=", e)
        tb = sys.exc_info()[2]
        print("LINE NO: ", tb.tb_lineno)

'''testimage="MI2.jpg"
prediction_image(testimage)'''

