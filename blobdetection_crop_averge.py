import cv2


def detect_blobs(frame, detector, roi_width, roi_height):
    keypoints = []

    height, width = frame.shape
    height, width = frame.shape
    center_roi = frame[height//3:2*height//3, width//3:2*width//3]
    roi_keypoints = detector.detect(center_roi)

    for kp in roi_keypoints:
        kp.pt = (kp.pt[0] + width//3, kp.pt[1] + height//3)

    keypoints.extend(roi_keypoints)

    return keypoints


def process_frame(frame, detector, prev_keypoints, moving_objects, total_objects):
    height, width = frame.shape[:2]
    roi_height, roi_width = height // 3, width // 3

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
    total_objects_per_frame = []
    moving_objects_per_frame = []

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
            keypoints = process_frame(frame, detector, prev_keypoints, moving_objects, total_objects)

            num_moving_objects = len(moving_objects)
            num_total_objects = len(total_objects)
            print("Moving objects: ", num_moving_objects)
            print("Total objects: ", num_total_objects)
            total_objects_per_frame.append(num_total_objects)
            moving_objects_per_frame.append(num_moving_objects)

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
    # print("Moving objects: ", num_moving_objects)
    # print("Total objects: ", num_total_objects)
    total_objects_per_frame.append(num_total_objects)
    moving_objects_per_frame.append(num_moving_objects)

    avg_total_objects_per_frame = sum(total_objects_per_frame) / len(total_objects_per_frame)
    avg_moving_objects_per_frame = sum(moving_objects_per_frame) / len(moving_objects_per_frame)

    # print("Average total objects per frame: ", avg_total_objects_per_frame)
    # print("Average moving objects per frame: ", avg_moving_objects_per_frame)

    return num_moving_objects, num_total_objects, avg_total_objects_per_frame, avg_moving_objects_per_frame

video_path = 'videos/1.mp4'
output_path = 'videos/1_video_crop.mp4'
result = process_video(video_path, output_path)

print("Moving objects: ", result[0])
print("Total objects: ", result[1])
print("Average total objects per frame: ", result[2])
print("Average moving objects per frame: ", result[3])