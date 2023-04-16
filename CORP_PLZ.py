import cv2
import os
import numpy as np

def enhance_contrast_brightness(frame, alpha, beta):
    return cv2.addWeighted(frame, alpha, frame, 0, beta)

def equalize_histogram(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    ycrcb = cv2.merge(channels)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def denoise(frame, h, templateWindowSize, searchWindowSize):
    return cv2.fastNlMeansDenoisingColored(frame, None, h, h, templateWindowSize, searchWindowSize)

def sharpen(frame, sigma):
    blurred_frame = cv2.GaussianBlur(frame, (0, 0), sigma)
    return cv2.addWeighted(frame, 1.5, blurred_frame, -0.5, 0)

def emphasize_head(frame, lower_threshold, upper_threshold):
    mask = cv2.inRange(frame, lower_threshold, upper_threshold)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result

def create_blob_detector(min_radius, max_radius):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = np.pi * min_radius**2
    params.maxArea = np.pi * max_radius**2
    params.minThreshold = 30
    params.filterByCircularity = True
    params.minCircularity = 0.3
    params.maxCircularity = 1.0
    params.filterByInertia = True
    params.minInertiaRatio = 0.15
    params.maxInertiaRatio = 1.0
    params.filterByConvexity = False

    return cv2.SimpleBlobDetector_create(params)

def is_moving(kp, prev_keypoints, threshold=3):
    for prev_kp in prev_keypoints:
        dx = abs(prev_kp.pt[0] - kp.pt[0])
        dy = abs(prev_kp.pt[1] - kp.pt[1])
        distance = np.sqrt(dx**2 + dy**2)
        if distance <= threshold:
            return False
    return True

def preprocess_frame(frame, target_width, target_height, apply_brightness_contrast=True, alpha=1, beta=20, enhance=True, 
                     equalize=False, denoise_flag=False, sharpen_flag=True, emphasize_head_flag=True):
    height, width = frame.shape[:2]

    center_roi = frame[height // 3:2 * height // 3, width // 3:2 * width // 3]

    if enhance:
        center_roi = enhance_contrast_brightness(center_roi, 1.2, 30)

    if equalize:
        center_roi = equalize_histogram(center_roi)

    if apply_brightness_contrast:
        frame = enhance_contrast_brightness(frame, alpha, beta)

    if denoise_flag:
        center_roi = denoise(center_roi, 3, 7, 21)

    if sharpen_flag:
        center_roi = sharpen(center_roi, 8)
    
    if emphasize_head_flag:
        lower_threshold = np.array([0, 0, 150])
        upper_threshold = np.array([255, 255, 255])
        center_roi = emphasize_head(center_roi, lower_threshold, upper_threshold)

    resized_color = cv2.resize(center_roi, (target_width, target_height))

    return resized_color

def corp_video_with_blob_detection(video_path, output_path, min_radius, max_radius, apply_brightness_contrast=True, alpha=1, beta=20, 
                                   enhance=True, equalize=False, denoise_flag=False, sharpen_flag=True, emphasize_head_flag=True):
    if not os.path.exists(video_path):
        print(f"Error: Input video file '{video_path}' does not exist.")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening the input video file")
        exit(1)
    else:
        print("Input video file opened successfully")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    target_width, target_height = 640, 480

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'X264'), fps, (target_width, target_height), isColor=True)

    if not out.isOpened():
        print("Error: Output video file could not be created. Check the codec and file path.")
        return

    detector = create_blob_detector(min_radius, max_radius)
    prev_keypoints = []
    
    frame_count = 0
    moving_objects = []
    total_objects = []
    num_moving_objects = 0
    num_total_objects = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now_frame = preprocess_frame(frame, target_width, target_height, apply_brightness_contrast, alpha, beta, enhance, equalize, denoise_flag, sharpen_flag)

        now_keypoints = detector.detect(now_frame)
        moving_kps = [kp for kp in now_keypoints if is_moving(kp, prev_keypoints)]
        # 디택션을 잡을 때 일단 다 잡고 나서 그 다음것들을 바운딩 박스로 그려서 그런거같다
        moving_objects.extend(moving_kps)
        total_objects.extend(now_keypoints)
        num_moving_objects += len(moving_kps)
        num_total_objects += len(now_keypoints)

        for kp in now_keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            radius = kp.size / 2

            if min_radius <= radius <= max_radius:
                size = int(kp.size)
                cv2.rectangle(now_frame, (x - size, y - size), (x + size, y + size), (0, 255, 0), 2)

        prev_keypoints = now_keypoints
        out.write(now_frame)

        cv2.imshow('Processed Frame', now_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if os.path.exists(output_path):
        print("영상 전처리 완료!")
        
    avg_moving_objects = num_moving_objects / frame_count
    avg_total_objects = num_total_objects / frame_count
    print("Moving objects: ", avg_moving_objects)
    print("Total objects: ", avg_total_objects)

    return avg_moving_objects, avg_total_objects
        
video_path = 'videos/9.mp4'
output_path = 'videos/real_last29.mp4'
min_radius = 3
max_radius = 7

corp_video_with_blob_detection(video_path, output_path, min_radius, max_radius)


