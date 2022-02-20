import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)  # only one face
id_list = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
plotY = LivePlot(640, 360, [0, 40], invert=True)
ratiolist = []
blinkCounter = 0
counter = 0

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # to let the camera on always

    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        for id in id_list:
            cv2.circle(img, face[id], 5, (255, 0, 255), thickness=cv2.FILLED)
        LeftUp = face[159]
        LeftDown = face[23]
        LeftLeft = face[130]
        LeftRight = face[243]

        lengthVert, _ = detector.findDistance(LeftUp, LeftDown)  # to get the number not the list  (the distance)
        lengthHor, _ = detector.findDistance(LeftLeft, LeftRight)

        cv2.line(img, LeftUp, LeftDown, (0, 200, 0), 3)  # to draw the distance between the leftup and the left down
        cv2.line(img, LeftLeft, LeftRight, (0, 200, 0),
                 3)  # to draw the distance between the left left  and the left right

        # print(int((lengthVert / lengthHor)*100))
        ratio = (lengthVert / lengthHor) * 100
        ratiolist.append(ratio)
        if len(ratiolist) > 3:
            ratiolist.pop(0)
        ratioAvg = sum(ratiolist) / len(ratiolist)

        # to calculate the number of blinks
        if ratioAvg < 26 and counter == 0:
            blinkCounter += 1
            counter = 1

        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0

        cvzone.putTextRect(img, f'blink count :{blinkCounter}', (50, 100))

        ImagePlot = plotY.update(ratioAvg)
        img = cv2.resize(img, (640, 360))

        imgStack = cvzone.stackImages([img, ImagePlot], 1, 1)  # plot them together


    else :
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, img], 1, 1)

    # img = cv2.resize(img, (640, 360))

    cv2.imshow('imgStack', imgStack)

    key = cv2.waitKey(1)
    if key == 27:
        break
