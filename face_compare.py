import cv2 as cv
import face_recognition as fr
from ipcamera import IPCamera
import inspect

ipcm = IPCamera()
ipcm.url = "rtsp://admin:09024792979Daniel@192.168.1.21/live/ch00_0"

img = cv.imread("assets/dwayne.png")
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_encoding = fr.face_encodings(img_rgb)[0]


img_t = cv.imread("assets/dwayne2.png")
img_rgb_t = cv.cvtColor(img_t, cv.COLOR_BGR2RGB)
img_encoding_t = fr.face_encodings(img_rgb_t)[0]

result = fr.compare_faces([img_encoding], img_encoding_t)
#print("Output: ", result)


ipcm.setCamera()