import cv2
# 현재 잡히는 객체의 정확도가 낮음/ 확인완료 있어야돼

def detect_blobs(frame, detector, roi_width, roi_height):
    keypoints = []

    height, width = frame.shape
    for i in range(3):
        for j in range(3):
            roi = frame[i * roi_height:(i + 1) * roi_height, j * roi_width:(j + 1) * roi_width]
            roi_keypoints = detector.detect(roi)

            for kp in roi_keypoints:
                kp.pt = (kp.pt[0] + j * roi_width, kp.pt[1] + i * roi_height)

            keypoints.extend(roi_keypoints)

    return keypoints


def process_frame(frame, detector, prev_keypoints, moving_objects, total_objects):
    height, width = frame.shape[:2]
    roi_height, roi_width = height // 3, width // 3

    if len(frame.shape) == 3:  # BGR 이미지일 경우
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:  # grayscale 이미지일 경우
        gray = frame

    keypoints = detect_blobs(gray, detector, roi_width, roi_height)

    moving_objects.extend([kp for kp in keypoints if not any([prev_kp.pt[0] - 3 <= kp.pt[0] <= prev_kp.pt[0] + 3 and prev_kp.pt[1] - 3 <= kp.pt[1] <= prev_kp.pt[1] + 3 for prev_kp in prev_keypoints])])
    total_objects.extend(keypoints)

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        size = int(kp.size)
        cv2.rectangle(frame, (x - size, y - size), (x + size, y + size), (0, 255, 0), 2)

    return keypoints

def process_video(video_path, output_path):
    print("Opening video file...")
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip = fps // 15

    target_width, target_height = 640, 480
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_width, target_height))

    # Blob detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 10

    detector = cv2.SimpleBlobDetector_create(params)

    moving_objects = []
    total_objects = []
    total_objects_per_frame = 0
    moving_objects_per_frame = 0

    print("Processing frames...")
    ret, prev_frame = cap.read()
    prev_frame = cv2.resize(prev_frame, (target_width, target_height))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    equalized_gray = cv2.equalizeHist(prev_gray)
    prev_keypoints = detector.detect(equalized_gray)
    total_objects.extend(prev_keypoints)
    frame_count = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            frame = cv2.resize(frame, (target_width, target_height))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            equalized_gray = cv2.equalizeHist(gray)
            blurred_gray = cv2.GaussianBlur(equalized_gray, (5, 5), 0)
            keypoints = process_frame(blurred_gray, detector, prev_keypoints, moving_objects, total_objects)

            total_objects_per_frame += len(keypoints)
            moving_objects_per_frame += len(moving_objects)

            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                size = int(kp.size)
                cv2.rectangle(frame, (x - size, y - size), (x + size, y + size), (0, 255, 0), 2)

            prev_keypoints = keypoints

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # 결과를 구하는 부분을 수정합니다.
    num_moving_objects = moving_objects_per_frame
    num_total_objects = total_objects_per_frame
    avg_total_objects_per_frame = num_total_objects / (frame_count - 1)
    avg_moving_objects_per_frame = num_moving_objects / (frame_count - 1)

    return num_moving_objects, num_total_objects, avg_total_objects_per_frame, avg_moving_objects_per_frame



# 이 부분은 기존 코드와 동일합니다.
video_path = 'videos/1.mp4'
output_path = 'videos/1_video_plz.mp4'
result = process_video(video_path, output_path)
print("Moving objects: ", result[0])
print("Total objects: ", result[1])
print("Average total objects per frame: ", result[2])
print("Average moving objects per frame: ", result[3])