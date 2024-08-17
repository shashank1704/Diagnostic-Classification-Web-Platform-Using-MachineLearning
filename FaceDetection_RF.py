
import os
from PIL import Image, ImageDraw
from face_recognition.face_recognition_cli import image_files_in_folder
import cv2
import pickle
import face_recognition




def load_model(model_file):

    with open(model_file, 'rb') as f:
        rf_model = pickle.load(f)

    return rf_model


def criminal_detection(testimage):

    model=load_model("RF_model.clf")

    test_image = face_recognition.load_image_file(testimage)

    face_locations = face_recognition.face_locations(test_image)

    if len(face_locations)>0:
        x_test = face_recognition.face_encodings(test_image)[0]
        prediction_result = model.predict([x_test])[0]
        print("Criminal Detection=",prediction_result)

        show_picture(prediction_result,face_locations,test_image)


    else:
        print("No Face Detected")




def show_picture(prediction_result,face_locations,test_image):


    pil_image = Image.fromarray(test_image).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for (top, right, bottom, left) in face_locations:

        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        criminal_name = prediction_result.encode("UTF-8")
        text_width, text_height = draw.textsize(criminal_name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), criminal_name, fill=(255, 255, 255, 255))

    del draw
    pil_image.show()


criminal_detection("15.jpg")