import cv2
import numpy as np


def detect_blobs(frame, detector):
    keypoints = detector.detect(frame)
    return keypoints


def preprocess_frame(frame, target_width, target_height):
    height, width = frame.shape[:2]
    center_frame = frame[height//3:2*height//3, width//3:2*width//3]
    resized_frame = cv2.resize(center_frame, (target_width, target_height))
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to improve contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    return gray


def process_frame(frame, detector, prev_frame=None):
    height, width = frame.shape[:2]

    if len(frame.shape) == 3:  # If frame is not grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:  # If frame is already grayscale
        gray = frame

    keypoints = detect_blobs(gray, detector)

    moving_objs = 0
    if prev_frame is not None:
        # Compute difference between current and previous frame
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)

        # Apply histogram equalization to improve contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        diff = clahe.apply(diff)

        # Apply threshold to create binary image
        threshold = 25
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Compute contours to detect moving objects
        try:
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except:
            _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Only consider contours with minimum area
            if cv2.contourArea(contour) > 50:
                moving_objs += 1

        # Apply canny edge detection to make contours more distinct
        canny = cv2.Canny(diff, 30, 150)

        # Combine canny edge image with thresholded image
        thresh = cv2.bitwise_or(thresh, canny)

    return keypoints, moving_objs, thresh


def process_video(video_path, output_path):
    print("Opening video file...")
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip = 5

    target_width, target_height = 640, 480
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_width, target_height))

    # Blob detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 10

    detector = cv2.SimpleBlobDetector_create(params)

    print("Processing frames...")
    frame_count = 0
    prev_frame = None
    total_objects = []
    prev_keypoints = []  # Add initialization for prev_keypoints
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            # Preprocess the frame and detect keypoints
            now_frame_raw = frame
            now_frame_gray = preprocess_frame(now_frame_raw, target_width, target_height)
            now_keypoints = detector.detect(now_frame_gray)

            # Compute moving objects using keypoint tracking
            moving_objects = []
            if prev_frame is not None:  # Only execute if not first frame
                for kp in now_keypoints:
                    # Check if the keypoint has a matching keypoint in the previous frame
                    if any([abs(prev_kp.pt[0] - kp.pt[0]) <= 5 and abs(prev_kp.pt[1] - kp.pt[1]) <= 5 for prev_kp in prev_keypoints]):
                        continue
                    moving_objects.append(kp)

            # Add current keypoints to total keypoints list
            total_objects.extend(now_keypoints)

            # Draw the detected keypoints
            for kp in now_keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                size = int(kp.size)
                cv2.rectangle(now_frame_gray, (x - size, y - size), (x + size, y + size), (0, 255, 0), 2)

            # Write the processed frame to the output video
            out.write(now_frame_gray)

            # Update previous keypoints and increment frame count
            prev_keypoints = now_keypoints
            cv2.imshow('Processed Frame', cv2.cvtColor(now_frame_gray, cv2.COLOR_GRAY2BGR))
            cv2.waitKey(1)
            frame_count += 1
        else:
            frame_count += 1
        prev_frame = frame  # Update previous frame

    cap.release()
    out.release()
    
    cv2.destroyAllWindows()

    num_moving_objects = len(moving_objects)
    num_total_objects = len(total_objects)
    avg_moving_objects = num_moving_objects / frame_count
    avg_total_objects = num_total_objects / frame_count
    print("Moving objects: ", avg_moving_objects)
    print("Total objects: ", avg_total_objects)

    return num_moving_objects, num_total_objects

video_path = 'videos/1.mp4'
output_path = 'videos/1_video_CORP.mp4'
moving_objects, total_objects = process_video(video_path, output_path)
