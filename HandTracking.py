import math

import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)

        self.mpDraw = mp.solutions.drawing_utils

        self.tipids = [4, 8, 12, 16, 20]

    def findHandPoints(self, frame, draw=True):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(hand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    # cv2.circle(frame, (cx, cy), 20, (255, 0, 255), cv2.FILLED)
                    cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

        return self.lmList

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipids[0]][1] > self.lmList[self.tipids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other Fingers
        for id in range(1, 5):
            if self.lmList[self.tipids[id]][2] < self.lmList[self.tipids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, frame, tip1, tip2, minDist=50, draw=True):
        x1, y1 = self.lmList[tip1][1:]
        x2, y2 = self.lmList[tip2][1:]
        cx, cy = int(x1 + x2) // 2, int(y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)

        if draw and length < minDist:
            cv2.circle(frame, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        return length, frame

def main():
    prevTime = 0
    currTime = 0
    vidCap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    detector = handDetector()

    while True:
        success, frame = vidCap.read()

        frame = detector.findHandPoints(frame)
        lmList = detector.findPosition(frame)

        if len(lmList) != 0:
            print(lmList[4])

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vidCap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()