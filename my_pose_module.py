import cv2
import mediapipe as mp
import numpy as np
import time



mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


class PoseDetection:
    def __init__(self, static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5):
        # self.static_image_mode = static_image_mode
        # self.model_complexity = model_complexity
        # self.enable_segmentation = enable_segmentation
        # self.min_detection_confidence = min_detection_confidence

        self.pose = mp_pose.Pose(static_image_mode=static_image_mode,
                            model_complexity=model_complexity,
                            enable_segmentation=enable_segmentation,
                            min_detection_confidence= min_detection_confidence)

    def get_pose(self, img, draw=True):
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.pose.process(frame_rgb)
        if self.result.pose_landmarks:
            if draw:
                mp_draw.draw_landmarks(img, self.result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        return img

    def get_positions(self, img, draw = False):
        positions = []
        if self.result.pose_landmarks:
            for id, landmark in enumerate(self.result.pose_landmarks.landmark):
                h,w, c = img.shape
                cy = int(landmark.y * h)
                cx = int(landmark.x * w)
                positions.append([id, cy, cx])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (0, 0, 255), -1)
        return positions








cap = cv2.VideoCapture("pose_videos/E.mp4")

def main():
    prev_time = 0
    pose_detection = PoseDetection()
    while True:
        success, frame = cap.read()
        cur_time = time.time()
        fps = int(1 / (cur_time - prev_time))
        prev_time = cur_time
        if not success:
            print("could not read frame")
            break
        img = pose_detection.get_pose(frame)
        positions =  pose_detection.get_positions(img)
        # print(positions)
        cv2.putText(img, str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()