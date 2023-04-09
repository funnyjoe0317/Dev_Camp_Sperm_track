import cv2
import os


def enhance_contrast_brightness(frame, alpha, beta):
    return cv2.addWeighted(frame, alpha, frame, 0, beta)

def equalize_histogram(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    ycrcb = cv2.merge(channels)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def enhance_contrast_brightness(frame, alpha, beta):
    return cv2.addWeighted(frame, alpha, frame, 0, beta)

def denoise(frame, h, templateWindowSize, searchWindowSize):
    return cv2.fastNlMeansDenoisingColored(frame, None, h, h, templateWindowSize, searchWindowSize)

def sharpen(frame, sigma):
    blurred_frame = cv2.GaussianBlur(frame, (0, 0), sigma)
    return cv2.addWeighted(frame, 1.5, blurred_frame, -0.5, 0)

def preprocess_frame(frame, target_width, target_height, apply_brightness_contrast=True, alpha=1.5, beta=50, enhance=False, equalize=False, denoise_flag=False, sharpen_flag=False):
    height, width = frame.shape[:2]

    center_roi = frame[height // 3:2 * height // 3, width // 3:2 * width // 3]
    
    if apply_brightness_contrast:
        frame = enhance_contrast_brightness(frame, alpha, beta)

    if enhance:
        center_roi = enhance_contrast_brightness(center_roi, 1.2, 30)

    if equalize:
        center_roi = equalize_histogram(center_roi)

    if denoise_flag:
        center_roi = denoise(center_roi, 3, 7, 21)

    if sharpen_flag:
        center_roi = sharpen(center_roi, 5)

    resized_color = cv2.resize(center_roi, (target_width, target_height))

    return resized_color


def corp_video(video_path, output_path, apply_brightness_contrast=True, alpha=1.5, beta=50, enhance=False, equalize=False, denoise_flag=False, sharpen_flag=False):
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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now_frame = preprocess_frame(frame, target_width, target_height, apply_brightness_contrast, alpha, beta, enhance, equalize, denoise_flag, sharpen_flag)

        out.write(now_frame)

        cv2.imshow('Processed Frame', now_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if os.path.exists(output_path):
        print("영상 전처리 완료!")
        
video_path = 'videos/9.mp4'
output_path = 'videos/17_output_enhance_equalize_denoise_flagTrue_sharpen_flag_h264.mp4'
corp_video(video_path, output_path,apply_brightness_contrast=True, alpha=2, beta=50, enhance=True, equalize=True, denoise_flag=False, sharpen_flag=True)
