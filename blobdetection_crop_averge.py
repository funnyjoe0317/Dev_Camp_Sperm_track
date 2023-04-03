import cv2
import numpy as np


def detect_blobs(frame, detector, roi_width, roi_height):
    keypoints = []

    height, width = frame.shape[:2]
    # center_roi = frame[height//3:2*height//3, width//3:2*width//3]
    roi_keypoints = detector.detect(frame)

    for kp in roi_keypoints:
        kp.pt = (kp.pt[0] + width//3, kp.pt[1] + height//3)

    keypoints.extend(roi_keypoints)

    return keypoints


def process_frame(frame, detector, prev_frame=None):
    height, width = frame.shape[:2]
    
    if len(frame.shape) == 3:  # If frame is not grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:  # If frame is already grayscale
        gray = frame

    keypoints = detect_blobs(gray, detector, width, height)

    moving_objs = 0
    if prev_frame is not None:
        # Compute difference between current and previous frame
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)

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

    return keypoints, moving_objs


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
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get the central part of the frame
        height, width = frame.shape[:2]
        center_frame = frame[height//3:2*height//3, width//3:2*width//3]

        if frame_count % frame_skip == 0:
            # Process the frame and get the keypoints and moving objects count
            resized_frame = cv2.resize(center_frame, (target_width, target_height))
            gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY) # Convert frame to grayscale
            keypoints, moving_objs = process_frame(gray, detector, prev_frame)

            # Draw the detected keypoints
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                size = int(kp.size)
                cv2.rectangle(resized_frame, (x - size, y - size), (x + size, y + size), (0, 255, 0), 2)

            # Write the processed frame to the output video
            out.write(resized_frame)

            # Print keypoints and moving objects count
            print(f"Frame {frame_count}: Keypoints={len(keypoints)}, Moving objects={moving_objs}")

        prev_frame = gray
        frame_count += 1

    cap.release()
    out.release()


video_path = 'videos/1.mp4'
output_path = 'videos/1_video_crop.mp4'
process_video(video_path, output_path)