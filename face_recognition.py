import cv2
import numpy as np

cam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
while True:
    ret, img = cam.read()

    img_copy = np.copy(img)
    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)



    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1);


    # go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv2.imshow('FaceDetection', img_copy)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
    print("\n close camera")
cam.release()
cv2.destroyAllWindows()