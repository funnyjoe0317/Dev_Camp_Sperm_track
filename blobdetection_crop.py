import cv2

def preprocess_frame(frame, target_width, target_height):
    height, width, _ = frame.shape
    roi_height, roi_width = height // 3, width // 3

    center_roi = frame[roi_height:2*roi_height, roi_width:2*roi_width]

    center_gray = cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY)
    
    equalized_gray = cv2.equalizeHist(center_gray)

    resized_gray = cv2.resize(equalized_gray, (target_width, target_height))

    return resized_gray, frame

def process_video(video_path, output_path):
    print("Opening video file...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening the input video file")
        exit(1)
    else:
        print("Input video file opened successfully") 
         
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip = fps // 15

    target_width, target_height = 640, 480
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_width, target_height))

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
    prev_frame_gray, prev_frame = preprocess_frame(prev_frame_raw, target_width, target_height)
    prev_keypoints = detector.detect(prev_frame_gray)
    total_objects.extend(prev_keypoints)
    frame_count = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            now_frame_raw = frame
            now_frame_gray, now_frame  = preprocess_frame(now_frame_raw, target_width, target_height)

            now_keypoints = detector.detect(now_frame_gray)
            moving_objects.extend([kp for kp in now_keypoints if not any([abs(prev_kp.pt[0] - kp.pt[0]) <= 5 and abs(prev_kp.pt[1] - kp.pt[1]) <= 5 for prev_kp in prev_keypoints])])
            total_objects.extend(now_keypoints)
            
            for kp in now_keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                size = int(kp.size)
                cv2.rectangle(now_frame, (x - size, y - size), (x + size, y + size), (0, 255, 0), 2)
            prev_keypoints = now_keypoints
            out.write(now_frame)
            cv2.imshow('Processed Frame', now_frame)
            cv2.waitKey(1)
            
            print(f'Frame {frame_count} processed')
        frame_count += 1

    cap.release()
    out.release()
    
    cv2.destroyAllWindows()

    num_moving_objects = len(moving_objects)
    num_total_objects = len(total_objects)
    avg_moving_objects = num_moving_objects / frame_count
    avg_total_objects = num_total_objects / frame_count    # 계속해서 갱신되는것인가 아니면 계속해서 추가되는 것인가
    print("Moving objects: ", avg_moving_objects)
    print("Total objects: ", avg_total_objects)

    return num_moving_objects, num_total_objects

video_path = 'videos/1.mp4'
output_path = 'videos/1_video.mp4'
moving_objects, total_objects = process_video(video_path, output_path)
# 무엇이 문제인제 모르겠지만 현재 바운딩 박스가 좌측 상단에 나타나고 객체를 분할한 상태에서 객체 검출이 아닌 전체 화면이 나오면서 거기서 검출을 하는중이다.