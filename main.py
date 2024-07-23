import cv2 as cv
import face_recognition as fr
from simple_facerec import SimpleFacerec

cap = cv.VideoCapture(0)
sfr = SimpleFacerec()
sfr.load_encoding_images("assets/")

while True:
    ret, frame = cap.read()

    # Detect faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x1, y2, x2 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv.putText(frame, name, (x2, y1 - 10), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv.imshow("Frame", frame)
    key = cv.waitKey(1)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()
