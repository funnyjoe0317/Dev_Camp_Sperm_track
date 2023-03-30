import cv2
import numpy as np


def preprocess_video_frame(frame, target_size, rotate_code=None):
    if rotate_code is not None:
        frame = cv2.rotate(frame, rotate_code)
    
    height, width, _ = frame.shape
    
    cell_height = height // 3
    cell_width = width // 3
    x1, y1 = cell_width, cell_height
    x2 = x1 + cell_width
    y2 = y1 + cell_height
    
    # Crop the center of the frame
    frame = frame[y1:y2, x1:x2]
    
    cropped_height, cropped_width, _ = frame.shape
    
    if cropped_height != target_size[1] or cropped_width != target_size[0]:
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return gray    

def process_video(video_path, output_path):
    print("Opening video file...")
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip = fps // 15
    
    # 영상 출력관련 부분 ---------------
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    # ---------------

    detector_params = cv2.SimpleBlobDetector_Params()
    # 필요한 경우 detector_params를 조정하세요.
    detector = cv2.SimpleBlobDetector_create(detector_params)

    moving_objects = []
    total_objects = []
    
    print("Processing frames...")
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_keypoints = detector.detect(prev_gray)
    total_objects.extend(prev_keypoints)

    frame_count = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            keypoints = detector.detect(gray)
            total_objects.extend(keypoints)

            for kp1 in prev_keypoints:
                for kp2 in keypoints:
                    if cv2.norm(kp1.pt, kp2.pt) <= 5:  # 임계값 설정
                        moving_objects.append(kp2)
                        break

            prev_keypoints = keypoints
            
            # 바운딩 박스 그리기
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                size = int(kp.size)
                cv2.rectangle(frame, (x - size, y - size), (x + size, y + size), (0, 255, 0), 2)
                
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    num_moving_objects = len(moving_objects)
    num_total_objects = len(total_objects)

    # return num_moving_objects, num_total_objects
    return print("Moving objects: ", num_moving_objects, "Total objects: ", num_total_objects)

if __name__ == "__main__":
    video_path = 'videos/1.mp4'
    output_path = 'videos/1_output.mp4'
    process_video(video_path, output_path)
