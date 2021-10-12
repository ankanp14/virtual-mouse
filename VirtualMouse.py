import cv2
import numpy as np
import HandTracking as htm
import time
import autopy

deviceIndex = 0
wCam, hCam = 1024, 768
frameRX = 200  # frame reduction
frameRY = 300  # frame reduction
smoothening = 4
clickDist = 40
clickDebounce = 1  # 1 second
dragInitDuration = 2  # 2 second

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

vidCap = cv2.VideoCapture(deviceIndex, cv2.CAP_DSHOW)
vidCap.set(3, wCam)
vidCap.set(4, hCam)

currTime = 0
prevTime = 0

prevClickTime = 0
isClickable = False
isDragable = False
isDragging = False

detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
print(wScr, hScr)

while True:
    success, frame = vidCap.read()
    frame = detector.findHandPoints(frame)
    lmList = detector.findPosition(frame, draw=False)

    currTime = time.time()

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # index finger at point 8
        x2, y2 = lmList[12][1:]  # middle finger at point 12
        x3, y3 = lmList[20][1:]  # finger at point 20

        fingers = detector.fingersUp()

        cv2.rectangle(frame, (frameRX, 0), (wCam - frameRX, hCam - frameRY), (255, 0, 255), 2)

        if fingers[1] == 1:
            x3 = np.interp(x1, (2*frameRX, wCam - frameRX), (0, wScr))
            y3 = np.interp(y1, (0, hCam - frameRY), (0, hScr))

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            try:
                if 0 < clocX < wScr and 0 < clocY < hScr:
                    autopy.mouse.move(wScr - clocX, clocY)
                else:
                    print("Out of bounds: ", clocX, clocY)
            except ValueError:
                print("Mouse point error: ", clocX, clocY)

            cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

            # find distance between thumb and point 6
            length, frame = detector.findDistance(frame, 8, 12, clickDist)
            plocX, plocY = clocX, clocY

            if fingers[4] == 1:
                if isDragging is False:
                    autopy.mouse.toggle(down=True)
                    isDragging = True
            else:
                if isDragging:
                    autopy.mouse.toggle(down=False)
                    isDragging = False

                if length < clickDist and isClickable is False:
                    isClickable = True

                if length > clickDist and isClickable:
                    autopy.mouse.click()
                    isClickable = False

    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(frame, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vidCap.release()
cv2.destroyAllWindows()