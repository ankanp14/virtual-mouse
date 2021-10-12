import cv2
import mediapipe as mp
import time

vidCap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

mpHands = mp.solutions.hands
hands = mpHands.Hands()

mpDraw = mp.solutions.drawing_utils

prevTime = 0
currTime = 0

while True:
    success, frame = vidCap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # cv2.circle(frame, (cx, cy), 20, (255, 0, 255), cv2.FILLED)
                cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vidCap.release()
cv2.destroyAllWindows()