# coding: UTF-8
import cv2
import numpy as np
import math
from math import sin, cos
from os import path
import camera

# cascades_dir = path.normpath(path.join(cv2.__file__, '..', '..', '..', '..', 'share', 'OpenCV', 'haarcascades'))
cascades_dir = "haarcascades"
max_size = 720

def detect(img):
    cascade_f = cv2.CascadeClassifier(path.join(cascades_dir, 'haarcascade_frontalface_alt2.xml'))
    cascade_e = cv2.CascadeClassifier(path.join(cascades_dir, 'haarcascade_eye.xml'))
    # resize if learch image
    rows, cols, _ = img.shape
    if max(rows, cols) > max_size:
        l = max(rows, cols)
        img = cv2.resize(img, (cols * max_size / l, rows * max_size / l))
    rows, cols, _ = img.shape
    # create gray image for rotate
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hypot = int(math.ceil(math.hypot(rows, cols)))
    frame = np.zeros((hypot, hypot), np.uint8)
    frame[int((hypot - rows) * 0.5):int((hypot + rows) * 0.5), int((hypot - cols) * 0.5):int((hypot + cols) * 0.5)] = gray
    # import pdb; pdb.set_trace()

    def translate(coord, deg):
        x, y = coord
        rad = math.radians(deg)
        return {
            'x': (  cos(rad) * x + sin(rad) * y - hypot * 0.5 * cos(rad) - hypot * 0.5 * sin(rad) + hypot * 0.5 - (hypot - cols) * 0.5) / float(cols) * 100.0,
            'y': (- sin(rad) * x + cos(rad) * y + hypot * 0.5 * sin(rad) - hypot * 0.5 * cos(rad) + hypot * 0.5 - (hypot - rows) * 0.5) / float(rows) * 100.0,
        }
    # rotate and detect faces
    results = []
    for deg in range(-48, 49, 6):
        M = cv2.getRotationMatrix2D((hypot * 0.5, hypot * 0.5), deg, 1.0)
        rotated = cv2.warpAffine(frame, M, (hypot, hypot))
        faces = cascade_f.detectMultiScale(rotated, 1.08, 2)
        print deg, len(faces)
        for face in faces:
            x, y, w, h = face
            # eyes in face?
            y_offset = int(h * 0.1)
            roi = rotated[y + y_offset: y + h, x: x + w]
            eyes = cascade_e.detectMultiScale(roi, 1.05)
            eyes = filter(lambda e: (e[0] > w / 2 or e[0] + e[2] < w / 2) and e[1] + e[3] < h / 2, eyes)
            if len(eyes) == 2 and abs(eyes[0][0] - eyes[1][0]) > w / 4:
                score = math.atan2(abs(eyes[1][1] - eyes[0][1]), abs(eyes[1][0] - eyes[0][0]))
                if eyes[0][1] == eyes[1][1]:
                    score = 0.0
                results.append({
                    'center': translate([x + w * 0.5, y + h * 0.5], -deg),
                    'w': float(w) / float(cols) * 100.0,
                    'h': float(h) / float(rows) * 100.0,
                    'eyes': [translate([x + e[0] + e[2] * 0.5, y + y_offset + e[1] + e[3] * 0.5], -deg) for e in eyes],
                    'score': score,
                })
    # unify duplicate faces
    faces = []
    for result in results:
        x, y = result['center']['x'], result['center']['y']
        exists = False
        for i in range(len(faces)):
            face = faces[i]
            if (face['center']['x'] - face['w'] * 0.5 < x < face['center']['x'] + face['w'] * 0.5 and
                face['center']['y'] - face['h'] * 0.5 < y < face['center']['y'] + face['h'] * 0.5):
                exists = True
                if result['score'] < face['score']:
                    faces[i] = result
                    break
        if not exists:
            faces.append(result)
    for face in faces:
        del face['score']
    return faces

if __name__ == "__main__":
    img = cv2.imread("tmp/images.jpg")
    faces = detect(img)
    # import pdb; pdb.set_trace()
    for face in faces:
        cx = int(face['center']['x'])
        cy = int(face['center']['y'])
        w = int(face['w'])
        h = int(face['h'])
        point1 = tuple([cx-w/2,cy-h/2])
        point2 = tuple([cx+w/2,cy+h/2])
        # point1 = tuple([cx,cy])
        # point2 = tuple([cx+w,cy+h])
        face_cut = img[point1[1]:point2[1],point1[0]:point2[0]]
        cv2.rectangle(img, point1, point2, (0,0,255), 1)
        cv2.namedWindow("face", cv2.WINDOW_NORMAL)
        cv2.imshow("face", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # frame = camera.face(img)
    # フレームを表示する(時間も計測する)
    # cv2.imshow("image", frame)
    cv2.imwrite("tmp/face_0001.jpg", img)
