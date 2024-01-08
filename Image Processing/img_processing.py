import cv2
import numpy as np
import os

frameWidth = 240
frameHeight = 360

imgPath = "D:/Projects/PyCharm/tinggiberatbadan/Machine Learning/dataset_img"
# img = cv2.imread(r"D:\Projects\PyCharm\tinggiberatbadan\Machine Learning\dataset_img\test.jpg")
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# success, img = cap.read()
# img = cv2.resize(img, (frameWidth, frameHeight))


def empty(a):
  pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 480, 144)
cv2.createTrackbar("Threshold1", "Parameters", 45, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 40, 255, empty)
cv2.createTrackbar("Area", "Parameters", 40, 2000, empty)

def stackImages(scale, imgArray):
  rows = len(imgArray)
  cols = len(imgArray[0])
  rowsAvailable = isinstance(imgArray[0], list)
  width = imgArray[0][0].shape[1]
  height = imgArray[0][0].shape[0]

  if rowsAvailable:
    for x in range(0, rows):
      for y in range(0, cols):
        if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
          imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
        else:
          imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale,
                                      scale)
        if len(imgArray[x][y].shape) == 2:
          imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)

    imageBlank = np.zeros((height, width, 3), np.uint8)
    hor = [imageBlank] * rows

    for x in range(0, rows):
      hor[x] = np.hstack(imgArray[x])

    ver = np.vstack(hor)
  else:
    for x in range(0, rows):
      if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
        imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
      else:
        imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
      if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)

    hor = np.hstack(imgArray)
    ver = hor

  return ver

def getContours(img, imgContour, areaContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > areaContour:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)

while True:
    success, img = cap.read()
    imgContour = img.copy()

    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    area = cv2.getTrackbarPos("Area", "Parameters")
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2, area)

    kernel = np.ones((3, 3), np.uint8)
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

    getContours(imgDil, imgContour, area)

    imgStack = stackImages(0.8, ([img, imgGray, imgCanny],
                                 [imgBlur, imgDil, imgContour]))

    cv2.imshow("result", imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
