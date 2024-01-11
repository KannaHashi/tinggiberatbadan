import cv2
import utlis
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression

###################################
webcam = True
path = '1.jpg'
cap = cv2.VideoCapture("http://172.20.10.2:81/stream")
# cap = cv2.VideoCapture(0)
cap.set(10,160)
cap.set(3,1024)
cap.set(4,768)
scale = 3
wP = 210 *scale
hP = 297 *scale
###################################

model = attempt_load("yolov5s.pt", map_location="cuda")

while True:
    if webcam:success,img = cap.read()
    else: img = cv2.imread(path)

    results = model(img)

    pred = non_max_suppression(results, conf_thres=0.5, iou_thres=0.4)[0]

    imgContours , conts = utlis.getContours(img,minArea=50000,filter=4)
    if len(conts) != 0:
        biggest = conts[0][2]
        #print(biggest)
        imgWarp = utlis.warpImg(img, biggest, wP,hP)
        imgContours2, conts2 = utlis.getContours(imgWarp,
                                                 minArea=2000, filter=4,
                                                 cThr=[30,30],draw = False)
        if len(conts) != 0:
            for obj in conts2:
                cv2.polylines(imgContours2,[obj[2]],True,(0,255,0),2)
                nPoints = utlis.reorder(obj[2])
                nW = round((utlis.findDis(nPoints[0][0]//scale,nPoints[1][0]//scale)/10),1)
                nH = round((utlis.findDis(nPoints[0][0]//scale,nPoints[2][0]//scale)/10),1)
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv2.putText(imgContours2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
                cv2.putText(imgContours2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
        cv2.imshow('A4', imgContours2)

    img = cv2.resize(img,(0,0),None,0.5,0.5)
    cv2.imshow('Original',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
