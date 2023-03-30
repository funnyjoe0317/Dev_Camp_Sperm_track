import os
import cv2

def create_lr_images(hr_dir, lr_dir, scale_factor=4):
    if not os.path.exists(lr_dir):
        os.makedirs(lr_dir)

    hr_image_files = os.listdir(hr_dir)

    for hr_image_file in hr_image_files:
        hr_image = cv2.imread(os.path.join(hr_dir, hr_image_file))
        height, width = hr_image.shape[:2]

        # Bicubic downsampling
        lr_image = cv2.resize(hr_image, (width // scale_factor, height // scale_factor), interpolation=cv2.INTER_CUBIC)

        # 저장할 파일 이름 설정
        file_name, ext = os.path.splitext(hr_image_file)
        lr_image_file = f"{file_name}_LR{ext}"

        cv2.imwrite(os.path.join(lr_dir, lr_image_file), lr_image)
        
def downsample_video(input_video_path, output_video_path, scale_factor=4):
    # 비디오 캡처 객체 생성
    video_capture = cv2.VideoCapture(input_video_path)

    # 비디오 속성 가져오기
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    
    # 다운샘플링된 비디오의 너비와 높이 계산
    new_width = width // scale_factor
    new_height = height // scale_factor

    # 비디오 쓰기 객체 생성
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (new_width, new_height))

    # 비디오 프레임 처리
    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if not ret:
            break

        # 프레임 다운샘플링
        downsampled_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # 다운샘플링된 프레임 저장
        video_writer.write(downsampled_frame)

    # 비디오 캡처 및 쓰기 객체 해제
    video_capture.release()
    video_writer.release()

# hr_dir = "path/to/your/hr_images"
# lr_dir = "path/to/your/lr_images"
# create_lr_images(hr_dir, lr_dir)

# 사용 예제
input_video_path = "videos/hr/250x.mp4"
output_video_path = "videos/lr/250_output_video.mp4"
scale_factor = 4

downsample_video(input_video_path, output_video_path, scale_factor)