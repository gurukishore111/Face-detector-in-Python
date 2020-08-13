import cv2
from random import randrange
# Loading the pre-trained model
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Choose the image to detect the face

img = cv2.imread('baby2.jpg')

# 0 choose the default webcam
webcam = cv2.VideoCapture(0)

# must  change to greyscale

grey_scaled_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces

face_coordinate = trained_face_data.detectMultiScale(grey_scaled_image)

# print(face_coordinate)

# Draw Rectangle
for (x, y, w, h) in face_coordinate:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(128, 256),
                                            randrange(256), randrange(256)), 2)

# Displaying image
cv2.imshow('Nanos Face Detector-Python', img)

cv2.waitKey()

print("Code Completed!")
