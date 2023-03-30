import cv2

def process_video(video_path, output_path):
    print("Opening video file...")
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip = fps // 15

    target_width, target_height = 640, 480
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_width, target_height))

    # Blob detector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 5
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.filterByConvexity = True
    params.minConvexity = 0.5
    params.filterByInertia = True
    params.minInertiaRatio = 0.5

    detector = cv2.SimpleBlobDetector_create(params)

    moving_objects = []
    total_objects = []

    print("Processing frames...")
    ret, prev_frame = cap.read()
    prev_frame = cv2.resize(prev_frame, (target_width, target_height))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_keypoints = detector.detect(prev_gray)
    total_objects.extend(prev_keypoints)
    frame_count = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            frame = cv2.resize(frame, (target_width, target_height))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 9분할 처리 및 blob detection
            height, width = gray.shape
            roi_height, roi_width = height // 3, width // 3

            keypoints = []
            for i in range(3):
                for j in range(3):
                    roi = gray[i * roi_height:(i + 1) * roi_height, j * roi_width:(j + 1) * roi_width]
                    roi_keypoints = detector.detect(roi)

                    for kp in roi_keypoints:
                        kp.pt = (kp.pt[0] + j * roi_width, kp.pt[1] + i * roi_height)
                    keypoints.extend(roi_keypoints)

            moving_objects.extend([kp for kp in keypoints if not any([prev_kp.pt[0] - 5 <= kp.pt[0] <= prev_kp.pt[0] + 5 and prev_kp.pt[1] - 5 <= kp.pt[1] <= prev_kp.pt[1] + 5 for prev_kp in prev_keypoints])])
            total_objects.extend(keypoints)
            
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                size = int(kp.size)
                cv2.rectangle(frame, (x - size, y - size), (x + size, y + size), (0, 255, 0), 2)
            prev_keypoints = keypoints
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
output_path = 'videos/video.mp4'
moving_objects, total_objects = process_video(video_path, output_path)
