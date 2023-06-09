import cv2
import numpy as np

def adjust_brightness(img, alpha=1.7, beta=0):
    adjusted_img = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)
    return adjusted_img

def preprocess_frame(frame, target_width, target_height, lower_threshold=20, gaussian_blur=(5, 5), alpha=0.8, beta=15):
    height, width = frame.shape[:2]

    center_roi = frame[height//3:2*height//3, width//3:2*width//3]

    center_gray = cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY)
    
    resized_gray = cv2.resize(center_gray, (target_width, target_height))
    
    # 가우시안 블러 적용
    blurred_gray = cv2.GaussianBlur(resized_gray, gaussian_blur, 0)
    
    equalized_gray = cv2.equalizeHist(blurred_gray)
    
    adjusted_gray = adjust_brightness(equalized_gray, alpha=alpha, beta=beta)
    
    # 픽셀 값이 lower_threshold보다 작은 경우 0으로 설정
    adjusted_gray[adjusted_gray < lower_threshold] = 0
    
    return adjusted_gray

def is_moving(kp, prev_keypoints, threshold=2):
    for prev_kp in prev_keypoints:
        dx = abs(prev_kp.pt[0] - kp.pt[0])
        dy = abs(prev_kp.pt[1] - kp.pt[1])
        distance = np.sqrt(dx**2 + dy**2)
        if distance <= threshold:
            return False
    return True

def process_video(video_path, output_path, min_radius, max_radius):
    # print("Opening video file...")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening the input video file")
        exit(1)
    else:
        print("Input video file opened successfully")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    target_width, target_height = 640, 480
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_width, target_height), isColor=False)

    if not out.isOpened():
        print("Error opening the output video file")
        exit(1)
    else:
        print("Output video file opened successfully")

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = np.pi * min_radius**2 
    # 면적 최소, 최대
    params.maxArea = np.pi * max_radius**2
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.maxCircularity = 1.0
    params.filterByInertia = True
    params.minInertiaRatio = 0.5
    params.maxInertiaRatio = 0.95

    detector = cv2.SimpleBlobDetector_create(params)

    moving_objects = []
    total_objects = []

    # print("Processing frames...")
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
        # print(f'움직이는 객체: {len(moving_objects)}')
        # print(f'전체 객체: {len(total_objects)}')

        # for kp in now_keypoints:
        #     x, y = int(kp.pt[0]), int(kp.pt[1])
        #     size = int(kp.size)
        #     cv2.rectangle(now_frame_gray, (x - size, y - size), (x + size, y + size), (0, 255, 0), 2)
        for kp in now_keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            radius = kp.size / 2
            
            if min_radius <= radius <= max_radius:
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
    avg_total_objects_37 = ((avg_total_objects/2)*1000)/0.00075
    print("Moving objects: ", avg_moving_objects)
    print("Moving %: ", (avg_moving_objects/avg_total_objects)*100)
    print("Total objects: ", avg_total_objects)
    print("Total objects_ALL: ", avg_total_objects_37)

    return avg_moving_objects, avg_total_objects

# video_path = 'videos/oview.mp4'
video_path = 'videos/sungmin.mp4'
output_path = 'videos/test025.mp4'
min_radius = 3.5
max_radius = 7
moving_objects, total_objects = process_video(video_path, output_path, min_radius, max_radius)