import cv2

def process_video(video_path, output_path):
    print("Opening video file...")
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip = fps // 15

    target_width, target_height = 640, 480
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_width, target_height))

    # Blob detector 이미지에서 영역을 찾는데 사용되는 알고리즘 
    params = cv2.SimpleBlobDetector_Params()
    # simpleblobdetector의 매개변수를 저장하는 객체
    params.filterByArea = True
    params.minArea = 10
    # params.filterByCircularity = True
    # params.minCircularity = 0.5
    # params.filterByConvexity = True
    # params.minConvexity = 0.5
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.5

    detector = cv2.SimpleBlobDetector_create(params)

    moving_objects = []
    total_objects = []

    print("Processing frames...")
    ret, prev_frame = cap.read()
    prev_frame = cv2.resize(prev_frame, (target_width, target_height))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    equalized_gray = cv2.equalizeHist(prev_gray)
    prev_keypoints = detector.detect(equalized_gray)
    # 그레이스케일 이미지에서 블롭(객체)를 검출합니다. 검출된 블롭의 정보는 prev_keypoints에 저장됩니다.
    total_objects.extend(prev_keypoints)
    # 검출된 블롭들을 total_objects 리스트에 추가합니다. 이렇게 하면 처리 과정에서 발견된 모든 블롭들을 한 곳에 모아 추적할 수 있습니다.
    frame_count = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            # 프레임 카운트가 frame_skip으로 나누어 떨어질 때만 블롭 검출을 수행합니다. 이렇게 하면 일정한 간격으로 프레임을 처리하게 됩니다.
            frame = cv2.resize(frame, (target_width, target_height))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 9분할 처리 및 blob detection
            height, width = gray.shape
            roi_height, roi_width = height // 3, width // 3

            keypoints = []
            # 이중 for 루프를 사용하여 9개의 서로 다른 영역에서 블롭 검출을 수행합니다. 각 영역에 대해 다음과 같은 작업을 수행합니다:
            for i in range(3):
                for j in range(3):
                    roi = gray[i * roi_height:(i + 1) * roi_height, j * roi_width:(j + 1) * roi_width]
                    # roi는 현재 영역의 이미지를 나타냅니다. 여기서 블롭 키 포인트는 블롭 중심의 좌표와 크기 정보를 포함합니다.
                    roi_keypoints = detector.detect(roi)
                    # 각 격자 영역에 대해 detector.detect(roi)를 사용하여 해당 영역의 블롭 키 포인트를 검출합니다.

                    for kp in roi_keypoints:
                        kp.pt = (kp.pt[0] + j * roi_width, kp.pt[1] + i * roi_height)
                        # roi_keypoints 리스트에 저장된 키포인트들의 좌표는 현재 ROI 영역의 상대좌표(relative coordinate) 값입니다. 이 좌표값을 변환하여, 전체 이미지 내에서의 절대좌표(absolute coordinate) 값으로 변경해야 합니다.
                        # ROI의 좌측 상단 꼭지점이 전체 이미지에서 어디에 위치하는지를 나타냅니다. roi_width와 roi_height는 각각 ROI의 너비와 높이를 나타내며, 이 값들을 이용하여 상대좌표값을 절대좌표값으로 변환합니다.
                    keypoints.extend(roi_keypoints)
                    # 검출된 각 블롭 키 포인트의 위치를 원래 이미지에 맞게 조정하고, keypoints 리스트에 추가합니다.

            moving_objects.extend([kp for kp in keypoints if not any([prev_kp.pt[0] - 5 <= kp.pt[0] <= prev_kp.pt[0] + 5 and prev_kp.pt[1] - 5 <= kp.pt[1] <= prev_kp.pt[1] + 5 for prev_kp in prev_keypoints])])
            # 여기가 제일 중요한데 이전 프레임에서 검출된 블롭 키 포인트와 현재 프레임에서 검출된 블롭 키 포인트가 일정 거리 이상 떨어져 있으면 moving_objects 리스트에 추가합니다.
            # 떨어져 있는 객체들을 담는 리스트입니다. 이 리스트에 담긴 객체들은 움직이고 있는 객체들입니다.
            total_objects.extend(keypoints)
            # 현재 프레임에서 검출된 블롭 키 포인트들을 total_objects 리스트에 추가합니다. 이 리스트에는 이전 프레임에서 검출된 블롭 키 포인트들도 포함되어 있습니다.
            
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                # 블롭 키 포인트의 위치를 나타내는 좌표값입니다. 중심점의 좌표값과 크기를 나타내는 size 변수를 이용하여 블롭을 표시합니다.
                size = int(kp.size)
                cv2.rectangle(frame, (x - size, y - size), (x + size, y + size), (0, 255, 0), 2)
            prev_keypoints = keypoints
            # 현재 프레임에서 검출된 블롭 키 포인트들을 prev_keypoints에 저장합니다. 다음 프레임에서 이전 프레임에서 검출된 블롭 키 포인트들과 비교하기 위함입니다.
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    num_moving_objects = len(moving_objects)
    num_total_objects = len(total_objects)
    print("Moving objects: ", num_moving_objects)
    print("Total objects: ", num_total_objects)

    return num_moving_objects, num_total_objects

video_path = 'videos/1.mp4'
output_path = 'videos/1_video.mp4'
moving_objects, total_objects = process_video(video_path, output_path)
