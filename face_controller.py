import cv2
import numpy as np

def blur_simples(image, factor=2):
    (h,w) = image.shape[:2]
    kh = int(h/factor)
    kw = int(w/factor)
    if kh%2 == 0:
        kh-=1
    if kw%2 == 0:
        kw-=1
    return cv2.GaussianBlur(image, (kh,kw), 0)

def blur_pixels(image, blocks=10):
    (h,w) = image.shape[:2]
    step_x = np.linspace(0,w, blocks+1, dtype=np.int32)
    step_y = np.linspace(0,h, blocks+1, dtype="int")

    for y in range(len(step_y)-1):
        for x in range(len(step_x)-1):
            sx = step_x[x]
            sy = step_y[y]
            ex = step_x[x+1]
            ey = step_y[y+1]

            (B, G, R) = cv2.mean(image[sy:ey, sx:ex])[:3]
            cv2.rectangle(image, (sx, sy), (ex,ey), (B, G, R), -1)
    return image

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
ret, img = cap.read()
(ih, iw) = img.shape[:2]
coord_x = np.linspace(0, iw, 4, dtype=np.int32)
coord_y = np.linspace(0, ih, 4, dtype=np.int32)
while ret:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 3)
    cv2.line(img, (coord_x[1], 0), (coord_x[1], ih) , (0,0,0), 2)
    cv2.line(img, (coord_x[2], 0), (coord_x[2], ih) , (0,0,0), 2)
    cv2.line(img, (0, coord_y[1]), (iw, coord_y[1]), (0,0,0), 2)
    cv2.line(img, (0, coord_y[2]), (iw, coord_y[2]), (0,0,0), 2)
    cv2.imshow("Video", img)
    ret, img = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()





























# End
