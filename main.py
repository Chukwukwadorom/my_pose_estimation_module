
from my_pose_module import PoseDetection as Pd
import cv2
import time

prev_time = 0
cap = cv2.VideoCapture("pose_videos/H.mp4")

pose_detection= Pd()
while True:
    success, frame = cap.read()
    cur_time = time.time()
    fps = int(1 / (cur_time - prev_time))
    prev_time = cur_time

    if not success:
        print("did not read frame")
        break

    img = pose_detection.get_pose(frame)
    positions = pose_detection.get_positions(img)
    # print(positions)
    cv2.putText(img, str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



