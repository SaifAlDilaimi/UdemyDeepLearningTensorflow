import cv2
import os
import numpy as np
import cv2
from time import sleep
from keras.models import Model, load_model
from AGenderMobileV2 import AGenderNetMobileV2


# statische vars
face_cascade_classifier = "haarcascade_frontalface_alt.xml"
video_file_path = "tes_1.mp4"
depth = 16
width = 8
face_size = 96

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

def crop_face(imgarray, section, margin=40, size=64):
    """
    :param imgarray: full image
    :param section: face detected area (x, y, w, h)
    :param margin: add some margin to the face detected area to include a full head
    :param size: the result image resolution with be (size x size)
    :return: resized image in numpy array with shape (size x size x 3)
    """
    img_h, img_w, _ = imgarray.shape
    if section is None:
        section = [0, 0, img_w, img_h]
    (x, y, w, h) = section
    margin = int(min(w,h) * margin / 100)
    x_a = x - margin
    y_a = y - margin
    x_b = x + w + margin
    y_b = y + h + margin
    if x_a < 0:
        x_b = min(x_b - x_a, img_w-1)
        x_a = 0
    if y_a < 0:
        y_b = min(y_b - y_a, img_h-1)
        y_a = 0
    if x_b > img_w:
        x_a = max(x_a - (x_b - img_w), 0)
        x_b = img_w
    if y_b > img_h:
        y_a = max(y_a - (y_b - img_h), 0)
        y_b = img_h
    cropped = imgarray[y_a: y_b, x_a: x_b]
    resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    resized_img = np.array(resized_img)
    return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)


def main():
    model = AGenderNetMobileV2()
    model.load_weights("model.10-3.8290-0.8965-6.9498.h5")
    face_cascade = cv2.CascadeClassifier(face_cascade_classifier)

    video_capture = cv2.VideoCapture(video_file_path)
    while not video_capture.isOpened():
        video_capture = cv2.VideoCapture(video_file_path)
        cv2.waitKey(1000)
        print("Wait for the header")

    # infinite loop, break by key ESC
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(32, 32)
        )

        print(faces)
        # placeholder for cropped faces
        face_imgs = np.empty((len(faces), face_size, face_size, 3))
        for i, face in enumerate(faces):
            face_img, cropped = crop_face(frame, face, margin=40, size=face_size)
            (x, y, w, h) = cropped
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
            face_imgs[i,:,:,:] = face_img
        if len(face_imgs) > 0:
            # predict ages and genders of the detected faces
            face_imgs = model.prep_image(face_imgs)
            results = model.predict(face_imgs)
            genders, age = model.decode_prediction(results)
            print(genders, age)
        # draw results
        for i, face in enumerate(faces):
            label = "{}, {}".format(int(age[i]),
                                    "F" if genders[i] == 0 else "M")
            draw_label(frame, (face[0], face[1]), label)

        cv2.imshow('Keras Faces', frame)
        if cv2.waitKey(5) == 27:  # ESC key press
            break
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()