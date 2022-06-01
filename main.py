import cv2
import numpy
from cv2 import threshold

cap = cv2.VideoCapture('motion_eye.mp4')

while True:
    ret, frame = cap.read()
    roi = frame[100:800, 200:1100]
    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # next row reduces the noise.
    gray_roi = cv2.GaussianBlur(gray_roi, (7,7), 0)

    _, threshold = cv2.threshold(gray_roi, 2, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        #cnt = numpy.array(cnt).reshape((-1,1,2)).astype(numpy.int32)
        #cv2.drawContours(roi, [cnt],-1, (0,0,255), 3)
        cv2.rectangle(roi, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)

        break

    cv2.imshow('threshold', threshold)
    cv2.imshow('Gray_roi', gray_roi)
    cv2.imshow('roi', roi)
    key = cv2.waitKey(28)
    if key == 27:
        break

cv2.destroyAllWindows()