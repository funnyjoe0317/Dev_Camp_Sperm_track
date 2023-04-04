import cv2
import numpy as np

def adjust_brightness(img, alpha=1.7, beta=0):
    adjusted_img = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)
    return adjusted_img

def preprocess_frame(frame, target_width, target_height):
    height, width = frame.shape[:2]

    # center_roi = frame[height//3:2*height//3, width//3:2*width//3]
    
    # 수정 중
    center_y, center_x = height // 2, width // 2
    half_width, half_height = target_width // 2, target_height // 2
    center_roi = frame[center_y - half_height:center_y + half_height, center_x - half_width:center_x + half_width]

    
    center_gray = cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY)
    
    
    # resized_gray = cv2.resize(center_gray, (target_width, target_height))
    
    # equalized_gray = cv2.equalizeHist(resized_gray)
    equalized_gray = cv2.equalizeHist(center_gray)
    
    adjusted_gray = adjust_brightness(equalized_gray, alpha=2, beta=3)
    return adjusted_gray

def is_moving(kp, prev_keypoints, threshold=5):
    for prev_kp in prev_keypoints:
        dx = abs(prev_kp.pt[0] - kp.pt[0])
        dy = abs(prev_kp.pt[1] - kp.pt[1])
        distance = np.sqrt(dx**2 + dy**2)
        if distance <= threshold:
            return False
    return True

def process_video(video_path, output_path, min_radius, max_radius):
    print("Opening video file...")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening the input video file")
        exit(1)
    else:
        print("Input video file opened successfully")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # target_width, target_height = 640, 480
    target_width, target_height = 480, 480
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_width, target_height), isColor=False)

    if not out.isOpened():
        print("Error opening the output video file")
        exit(1)
    else:
        print("Output video file opened successfully")

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = np.pi * min_radius**2
    params.maxArea = np.pi * max_radius**2
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.maxCircularity = 1.0
    params.filterByInertia = True
    params.minInertiaRatio = 0.7
    params.maxInertiaRatio = 1.0

    detector = cv2.SimpleBlobDetector_create(params)

    moving_objects = []
    total_objects = []

    print("Processing frames...")
    ret, prev_frame_raw = cap.read()
    if not ret:
        print("Error reading the first frame")
        return
    prev_frame_gray = preprocess_frame(prev_frame_raw, target_width, target_height)
    prev_keypoints = detector.detect(prev_frame_gray)
    total_objects.extend(prev_keypoints)
    out.write(prev_frame_gray)
    frame_count = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now_frame_raw = frame
        now_frame_gray = preprocess_frame(now_frame_raw, target_width, target_height)

        now_keypoints = detector.detect(now_frame_gray)
        moving_kps = [kp for kp in now_keypoints if is_moving(kp, prev_keypoints)]

        moving_objects.extend(moving_kps)
        total_objects.extend(now_keypoints)
        print(f'움직이는 객체: {len(moving_objects)}')
        print(f'전체 객체: {len(total_objects)}')

        for kp in now_keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            size = int(kp.size)
            cv2.rectangle(now_frame_gray, (x - size, y - size), (x + size, y + size), (0, 255, 0), 2)
        
        prev_keypoints = now_keypoints
        out.write(now_frame_gray)

        cv2.imshow('Processed Frame', cv2.cvtColor(now_frame_gray, cv2.COLOR_GRAY2BGR))
        cv2.waitKey(1)
        print(f'Frame {frame_count} processed')

        frame_count += 1

    cap.release()
    out.release()

    num_moving_objects = len(moving_objects)
    num_total_objects = len(total_objects)
    avg_moving_objects = num_moving_objects / frame_count
    avg_total_objects = num_total_objects / frame_count
    print("Moving objects: ", avg_moving_objects)
    print("Total objects: ", avg_total_objects)

    return num_moving_objects, num_total_objects

video_path = 'videos/oview.mp4'
output_path = 'videos/oview_fir_start_onlymoving12.mp4'
min_radius = 3.5
max_radius = 8.5
moving_objects, total_objects = process_video(video_path, output_path, min_radius, max_radius)