import cv2
import numpy as np
# 프레임 압축버전

def adjust_brightness(img, alpha=1.7, beta=0):
    adjusted_img = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)
    return adjusted_img

def preprocess_frame(frame, target_width, target_height):
    height, width = frame.shape[:2]

    center_roi = frame[height//3:2*height//3, width//3:2*width//3]

    center_gray = cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY)
    
    resized_gray = cv2.resize(center_gray, (target_width, target_height))
    
    equalized_gray = cv2.equalizeHist(resized_gray)
    
    adjusted_gray = adjust_brightness(equalized_gray, alpha=2, beta=3)
    return adjusted_gray


def process_video(video_path, output_path):
    print("Opening video file...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening the input video file")
        exit(1)
    else:
        print("Input video file opened successfully") 
         
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # frame_skip = fps // 15

    target_width, target_height = 640, 480
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_width, target_height), isColor=False)

    if not out.isOpened():
        print("Error opening the output video file")
        exit(1)
    else:
        print("Output video file opened successfully")
        
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 10

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
        # if frame_count % frame_skip == 0:
        #     now_frame_raw = frame
        #     now_frame_gray = preprocess_frame(now_frame_raw, target_width, target_height)

        #     now_keypoints = detector.detect(now_frame_gray)
        #     moving_objects.extend([kp for kp in now_keypoints if not any([abs(prev_kp.pt[0] - kp.pt[0]) <= 3 and abs(prev_kp.pt[1] - kp.pt[1]) <= 3 for prev_kp in prev_keypoints])])
        #     total_objects.extend(now_keypoints)
            
        #     for kp in now_keypoints:
        #         x, y = int(kp.pt[0]), int(kp.pt[1])
        #         size = int(kp.size)
        #         cv2.rectangle(now_frame_gray, (x - size, y - size), (x + size, y + size), (0, 255, 0), 2)
        #     prev_keypoints = now_keypoints
        #     out.write(now_frame_gray)
        #     # 여기서 담기는 건데 왜 안담길까
        #     cv2.imshow('Processed Frame', cv2.cvtColor(now_frame_gray, cv2.COLOR_GRAY2BGR))
        #     cv2.waitKey(1)
            # ------------------

        now_frame_raw = frame
        now_frame_gray = preprocess_frame(now_frame_raw, target_width, target_height)

        now_keypoints = detector.detect(now_frame_gray)
        moving_objects.extend([kp for kp in now_keypoints if not any([abs(prev_kp.pt[0] - kp.pt[0]) <= 1 and abs(prev_kp.pt[1] - kp.pt[1]) <= 1 for prev_kp in prev_keypoints])])
        total_objects.extend(now_keypoints)
        
        for kp in now_keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            size = int(kp.size)
            cv2.rectangle(now_frame_gray, (x - size, y - size), (x + size, y + size), (0, 255, 0), 2)
        prev_keypoints = now_keypoints
        out.write(now_frame_gray)
        # 여기서 담기는 건데 왜 안담길까
        cv2.imshow('Processed Frame', cv2.cvtColor(now_frame_gray, cv2.COLOR_GRAY2BGR))
        cv2.waitKey(1)
        print(f'Frame {frame_count} processed')
            # print(f'Frame {frame_count} processed')
        frame_count += 1

    cap.release()
    out.release()
    
    # cv2.destroyAllWindows()

    num_moving_objects = len(moving_objects)
    num_total_objects = len(total_objects)
    avg_moving_objects = num_moving_objects / frame_count
    avg_total_objects = num_total_objects / frame_count    # 계속해서 갱신되는것인가 아니면 계속해서 추가되는 것인가
    print("Moving objects: ", avg_moving_objects)
    print("Total objects: ", avg_total_objects)

    return num_moving_objects, num_total_objects



video_path = 'videos/1.mp4'
output_path = 'videos/1_frame_video_CORP_alpa2_beta_3_moveupdate.mp4'
moving_objects, total_objects = process_video(video_path, output_path)
