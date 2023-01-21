import cv2
import mediapipe as mp
import numpy as np
import time



mp_pose = mp.solutions.pose
draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5)

fps = 60
cap = cv2.VideoCapture("pose_videos/H.mp4")

prev_time = 0
cur_time = 0

while True:
    success, frame =  cap.read()
    cur_time = time.time()
    fps = int(1 / (cur_time - prev_time))
    prev_time = cur_time
    if not success:
        print("could not read frame")
        break
    else:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)
        if not result.pose_landmarks:
            continue
        for id, landmark in enumerate(result.pose_landmarks.landmark):
            h,w, c = frame.shape
            cy = int(landmark.y * h)
            cx = int(landmark.x * w)

            print(id, cy, cx)
            #
            # if id == 4:
            #     cv2.circle(frame, (cx,cy), 15, (0,0, 255), -1)

        draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(frame, str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()