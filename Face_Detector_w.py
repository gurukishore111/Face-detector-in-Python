import cv2
from random import randrange
# Loading the pre-trained model
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')


# 0 choose the default webcam
webcam = cv2.VideoCapture(0)


# must  change to greyscale
while True:

    successfull_frame_read, frame = webcam.read()

    grey_scaled_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces

    face_coordinate = trained_face_data.detectMultiScale(grey_scaled_image)

    # print(face_coordinate)

    # Draw Rectangle
    for (x, y, w, h) in face_coordinate:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(128, 256),
                                                  randrange(256), randrange(256)), 2)

    # Displaying webcam
    cv2.imshow('Nanos Face Detector-Python', frame)

    key = cv2.waitKey(1)

    # Stop when Qkry is pressed

    if key == 81 or key == 113:
        break

# Webcam Stop

webcam.release()

print("Job Completed!")
