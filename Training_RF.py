import os
from face_recognition.face_recognition_cli import image_files_in_folder
import pickle
import face_recognition

from sklearn.ensemble import RandomForestClassifier




def get_facial_features(dataset_path):

    x_features = []
    target_features = []


    for class_dir in os.listdir(dataset_path):
        for image in image_files_in_folder(os.path.join(dataset_path, class_dir)):
            face = face_recognition.load_image_file(image)

            faces = face_recognition.face_locations(face)
            if len(faces)>0:
                x_features.append(face_recognition.face_encodings(face)[0])
                target_features.append(class_dir)


    return  x_features,target_features


def train_RF_model():

    x_train,y_train=get_facial_features("../FaceDetection/dataset")

    rf_clf=RandomForestClassifier(n_estimators=80,max_depth = 10)
    rf_clf.fit(x_train,y_train)

    with open("RF_model.clf", 'wb') as f:
        pickle.dump(rf_clf, f)

    print("RandomForest model Created..!")



train_RF_model()




