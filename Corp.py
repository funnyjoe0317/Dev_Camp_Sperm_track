import cv2
import os

def preprocess_frame(frame, target_width, target_height):
    height, width = frame.shape[:2]

    center_roi = frame[height//3:2*height//3, width//3:2*width//3]

    resized_color = cv2.resize(center_roi, (target_width, target_height))

    return resized_color

def corp_video(video_path, output_path):
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
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_width, target_height), isColor=True)

    if not out.isOpened():
        print("Error: Output video file could not be created. Check the codec and file path.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now_frame = preprocess_frame(frame, target_width, target_height)
        
        out.write(now_frame)
        
    cap.release()
    out.release()

    if os.path.exists(output_path):
        print("영상 전처리 완료!")
    else:
        print("Error: Output video file was not created.")

video_path = 'videos/13.mp4'
output_path = 'videos/17_output.mp4'
corp_video(video_path, output_path)
