import cv2
import json
import torch
import numpy as np
from sort import Sort
from motility_analysis import analyze_motility
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 나중을 위해서 함수로 만들어 놓는다.
def preprocess_video_frame(frame, target_size, rotate_code=None):
    if rotate_code is not None:
        frame = cv2.rotate(frame, rotate_code)
    # print(target_size)

    height, width, _ = frame.shape # 3번째는 1을 하면 그레이 스케일로 바뀜 이것을 사용해야할까?
    
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

def detect_and_track_sperms(video_path, output_video_path, target_size):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error:video file could not be opened")
        return
    
    tracks = {}
    frame_count = 0

    model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt",force_reload=True)
    model.conf = 0.1
    
    print("model loaded successfully")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 출력 동영상의 코덱을 설정합니다. mp4v는 MPEG-4 코덱을 의미합니다.
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Frame Per Second (FPS)를 구합니다.
    out = cv2.VideoWriter(output_video_path, fourcc, fps, target_size)

    tracker = Sort()
    #SORT (Simple Online and Realtime Tracking) 알고리즘은 다수의 프레임에서 객체를 추적하는 알고리즘
    
    # accumulated_distances = {}
    previous_positions = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Reached the end of the video")
            break
        # Preprocess the frame : crop the center and convert to grayscale
        preprocessed_frame = preprocess_video_frame(frame, target_size, rotate_code=None)

        results = model(preprocessed_frame)
        # 객체 탐지 모델을 적용하여서 결과를 받아온다. 나는 이 결과를 사용해서 객체를 추적할 것이다. 나는 yolov5를 사용했다.
        detections = results.xyxy[0].cpu().numpy() # 여기서 받은 첫 객체의 수를 저장해서 나중에 사용할 것이다.
        # results 객체에서 감지된 객체의 경계 상자, 신뢰도 점수 및 클래스 ID를 추출
        # xyxy 속성에는 각 경계 상자의 왼쪽 위 및 오른쪽 아래 모서리의 좌표가 포함, cpu() 메서드는 결과를 GPU 메모리에서 CPU 메모리로 이동시키고 numpy()는 결과를 나중에 쉽게 조작하기 위해 NumPy 배열로 변환

        # 출력 결과= detections 를 Sort에 적용하기 위해 다음과 같은 형식으로 변환
        detection_for_sort = []
        for detection in detections:
            # cls == 0은 객체가 탐지되지 않았을 때  
            x1, y1, x2, y2, conf, cls = detection
            if cls == 0:
                # cls 가 0일 경우에 객체가 검출되지 않았으므로 그것들을 detection_for_sort에 추가(쓰레기통 개념)
                detection_for_sort.append([x1, y1, x2, y2, conf])
                

        # Update SORT tracker
        if len(detection_for_sort) > 0:
            tracked_objects = tracker.update(np.array(detection_for_sort))
            # tracked_objects =[x1, y1, x2, y2, track_id] 
        else:
            tracked_objects = np.empty((0, 5))

        frame_tracks = []
        # frame_tracks = [x, y, track_id]
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track
            x, y = (x1 + x2) / 2, (y1 + y2) / 2
            # 평균을 구해서 중심점을 구한다.
            frame_tracks.append([x, y, track_id])
            # 평균 좌표를 구해서 frame_tracks에 추가한다.

            # 중심점을 그린다.
            cv2.circle(preprocessed_frame, (int(x), int(y)), 3, (0, 255, 0), 2)
            
            if track_id in previous_positions: # 이 부분이 이상해서 안그려져
                # 이전 위치가 있으면 이전 위치와 현재 위치를 연결한다.
                prev_x, prev_y = previous_positions[track_id]
                # 이전 위치를 저장한다.
                cv2.line(preprocessed_frame, (int(prev_x), int(prev_y)), (int(x), int(y)), (0, 255, 0), 2)
                # 이전 위치와 현재 위치를 연결한다.
                
            previous_positions[track_id] = (x, y)

        tracks[f"frame_{frame_count}"] = frame_tracks
        frame_count += 1

        output_frame = cv2.cvtColor(preprocessed_frame, cv2.COLOR_GRAY2BGR)
        out.write(output_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("detect_and_track_sperms function completed.")
    return tracks

# 여기서 영상만들 저장하고 실행 할 수 있다. 
def main_function():
    # sperm = "1"
    # sperm = "21_sperm_sample"
    sperm = "600_sample"
    video_path = f'videos/{sperm}.mp4'
    output_video_path = f"videos/{sperm}_output1.mp4"
    tracks = detect_and_track_sperms(video_path, output_video_path,(640, 480))

    with open("Detections_test_tracks.json", "w") as fp:
        json.dump(tracks, fp)

    with open("Detections_test_tracks.json", "r") as fp:
        tracks = json.load(fp)

    motility_results, sperm_counts = analyze_motility(tracks)

    # Print the results
    print("Motility Results:")
    for video_name, motility_data in motility_results.items():
        print(f"Video: {video_name}")
        for category, value in motility_data.items():
            print(f"{category}: {value}%")
        print()

    print("Sperm Counts:") 
    for video_name, count in sperm_counts.items():
        print(f"Video: {video_name}, Count: {count}")
    

if __name__ == "__main__":
    main_function()