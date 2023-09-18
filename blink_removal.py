import argparse
import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist

# Set print options to suppress scientific notation
np.set_printoptions(threshold=np.inf, suppress=True)

def calculate_eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2 * C)
    return ear

def main():
    # Parse command-line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
    ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
    args = vars(ap.parse_args())

    # Load the video
    print("[INFO] Loading Video")
    cap = cv2.VideoCapture(args["video"])

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Total number of frames: {total_no_frames}, fps: {fps}, duration: {total_no_frames / fps} seconds")

    # Constants
    EYE_BLINK_THRESH = 0.25
    SUCC_FRAME = fps / 10
    COUNTER = 0
    TOTAL = 0

    # Load the pre-trained facial landmark detector
    print("[INFO] Loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    landmark_predict = dlib.shape_predictor(args["shape_predictor"])

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    blink_frame_list = []

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(frame_gray)

        if faces:
            face = faces[0]
            shape = landmark_predict(frame_gray, face)
            shape = face_utils.shape_to_np(shape)

            left_eye = shape[lStart:lEnd]
            right_eye = shape[rStart:rEnd]

            left_EyeHull = cv2.convexHull(left_eye)
            right_EyeHull = cv2.convexHull(right_eye)

            cv2.drawContours(frame, [left_EyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_EyeHull], -1, (0, 255, 0), 1)

            left_EAR = calculate_eye_aspect_ratio(left_eye)
            right_EAR = calculate_eye_aspect_ratio(right_eye)

            avg_EAR = (left_EAR + right_EAR) / 2

            if avg_EAR < EYE_BLINK_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= SUCC_FRAME and COUNTER < 10:
                    pass
                if COUNTER >= SUCC_FRAME:
                    TOTAL += 1
                    for i in range(COUNTER, 0, -1):
                        blink_frame = cap.get(cv2.CAP_PROP_POS_FRAMES) - i
                        blink_frame_list.append(blink_frame)

                COUNTER = 0

            cv2.putText(frame, "Blink Counter: {}".format(TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(avg_EAR), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Image", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    blink_frame_array = np.array(blink_frame_list)
    all_frame_array = np.arange(1, total_no_frames + 1)
    mask = np.isin(all_frame_array, blink_frame_array, invert=True)
    no_blink_frame_array = all_frame_array[mask]

    cap.release()
    cv2.destroyAllWindows()

    print("[INFO] COMPLETED EXTRACTION OF FRAMES WITH BLINKS")

    # Write unblinked frames to another video
    print("[INFO] Writing out new video without blinking frames")

    # Output video file
    output_video_path = 'unblink_output_video.mp4'

    # Open the input video
    cap = cv2.VideoCapture(args["video"])

    # Define the codec and create VideoWriter object for the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_number = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_number += 1

        if frame_number in no_blink_frame_array:
            out.write(frame)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
